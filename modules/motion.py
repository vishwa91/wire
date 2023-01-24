#!/usr/bin/env python

'''
    Subroutines for estimating motion between images
'''

import os
import sys
import tqdm
import pdb
import math

import numpy as np
from scipy import linalg
from scipy import interpolate

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import kornia

from pystackreg import StackReg

import cv2

class ImageSRDataset(Dataset):
    def __init__(self, imstack, Xstack=None, Ystack=None, masks=None,
                 jitter=False, xjitter=None, yjitter=None, get_indices=False):
        super().__init__()
        
        self.imstack = imstack
        self.Xstack = Xstack
        self.Ystack = Ystack
        self.masks = masks
        self.jitter = jitter
        self.get_indices = get_indices
        
        self.nimg, _, self.H, self.W = imstack.shape
        
        if xjitter is None:
            self.xjitter = 1/self.W
            self.yjitter = 1/self.H
        else:
            self.xjitter = xjitter
            self.yjitter = yjitter

    def __len__(self):
        return self.nimg

    def __getitem__(self, idx):
        img = torch.tensor(self.imstack[idx, ...])
        
        # If Jitter is enabled, return stratified sampled coordinates
        pixels = img[None, ...].permute(0, 2, 3, 1).view(-1, 3)
        
        if self.masks is not None:
            mask = torch.tensor(self.masks[idx, ...])
            mask = mask[None, ...].permute(0, 2, 3, 1).view(-1, 3)
        else:
            mask = torch.zeros(1)
            
        if self.Xstack is not None:
            coords = torch.stack((torch.tensor(self.Xstack[idx, ...]),
                              torch.tensor(self.Ystack[idx, ...])),
                             dim=-1).reshape(-1, 2)
        else:
            coords = torch.zeros(1)
        
        if self.get_indices:
            return coords, pixels, mask, idx
        else:
            return coords, pixels, mask

def xy_mgrid(H, W):
    '''
        Generate a flattened meshgrid for heterogenous sizes
        
        Inputs:
            H, W: Input dimensions
        
        Outputs:
            mgrid: H*W x 2 meshgrid
    '''
    Y, X = torch.meshgrid(torch.linspace(-1, 1, H),
                          torch.linspace(-1, 1, W))
    mgrid = torch.stack((X, Y), dim=-1).reshape(-1, 2)
    
    return mgrid

def getEuclidianMatrix(theta, shift):
    '''
        Compute 2x3 euclidean matrix
    '''
    mat = np.array([[np.cos(theta), -np.sin(theta), shift[0]],
                    [np.sin(theta),  np.cos(theta), shift[1]]])
    
    return mat

def fb_flow(frame1, frame2):
    H, W = frame1.shape
    Y, X = np.mgrid[:H, :W]
    
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[...,1] = 255
    flow = cv2.calcOpticalFlowFarneback(frame1, 
                                        frame2,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    frame2_warped = cv2.remap(frame2.astype(np.float32),
                              (X + flow[..., 0]).astype(np.float32),
                              (Y + flow[..., 1]).astype(np.float32),
                              cv2.INTER_LINEAR)
    
    rgb_comp = np.zeros((H, W, 3))
    rgb_comp[..., 0] = frame1
    rgb_comp[..., 2] = frame2_warped

    return frame2_warped, flow, rgb, rgb_comp

def ecc_flow(im1, im2, warp_mode=cv2.MOTION_HOMOGRAPHY, niters=1000, eps=-1):
    '''
        Register images using Opencv intensity based image alignment approach.
        
        Inputs:
            im1, im2: Images to register. im2 will be registered to im1.
            method: One of cv2.MOTION_*** . Default is MOTION_HOMOGRAPRHY
            niters: Number of ECC iterations
            eps: Stopping tolerance
            
        Outputs:
            warp_matrix: Warping matrix
            im2_aligned: Second image warped to first image's coordinates
            flow: Flow coordinates to go from im2 to im1
            
        https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    '''
    # Find size of image1
    sz = im1.shape

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, niters, eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, 
                                             criteria=criteria, inputMask=None,
                                             gaussFiltSize=5)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1],sz[0]), 
                                          flags=flags)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]),
                                     flags=flags)
        
    # Create flow coordinates
    Y, X = np.mgrid[:sz[0], :sz[1]]
    coords = np.ones((3, sz[0]*sz[1]))
    coords[0, :] = X.reshape(1, -1)
    coords[1, :] = Y.reshape(1, -1)
    
    coords_new = warp_matrix.dot(coords)
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        coords_new = coords_new[:2, :]/coords_new[2, :]
    
    flow = np.zeros((sz[0], sz[1], 2), dtype=np.float32)
    flow[..., 0] = (coords_new[0, :] - coords[0, :]).reshape(sz)*(2/sz[1])
    flow[..., 1] = (coords_new[1, :] - coords[1, :]).reshape(sz)*(2/sz[0])
    
    return warp_matrix, im2_aligned, flow

def get_SR_data(im, scale, nimg=10, simulation=True, get_gt=False, shift_max=10,
                theta_max=np.pi/12, downsample=False):
    '''
        Wrapper function to get real or simulation data
        
        Inputs:
            im: Image or image stack
            scale: Scale for resolution
            nimg: Number of images
            simulation: If True, im will be converted to an image stack, else
                the input will be treated as imstack
            get_gt: If True, and simulation is also True, return groundtruth
                registration matrices
            shift_max, theta_max: See get_imstack
            downsample: If True, the imstack will be a downsampled version of
                the data. Only applicable if simulation is false
        Outputs:
            im: Ground truth high resolution image. Only useful if simulation
                is true, or simulation is false, and downsample is true. Else
                it is just a nearest neighbor upsampling
            imstack: (nimg, Hl, Wl) stack of low resolution images
            ecc_mats: (nimg, 2, 3) affine matrices
    '''
    if simulation is True:
        H, W = im.shape
        imstack, _, _, mats = get_imstack(im, scale, shift_max,
                                                    theta_max, nimg)
        _, Hl, Wl = imstack.shape
        if get_gt:
            ecc_mats = invert_regstack(mats)
        else:
            ecc_mats = register_stack(imstack, (Hl, Wl))[:, :2, :]
    else:
        _, H, W = im.shape
        imstack = np.copy(im[:nimg, ...], order='C')
        interp = cv2.INTER_LINEAR
        if downsample:
            imstack_lr = np.zeros((nimg, H//scale, W//scale))
            Hl, Wl = H//scale, W//scale
            for idx in range(nimg):
                imstack_lr[idx, ...] = cv2.resize(imstack[idx, ...],
                                                  (W//scale, H//scale),
                                                  interpolation=interp)
            im = imstack[0, ...]
            imstack = imstack_lr.astype(np.float32)
        else:
            im = cv2.resize(imstack[0, ...], (W*scale, H*scale))
            Hl, Wl = H, W
            H, W = Hl*scale, Wl*scale
        
        ecc_mats = register_stack(imstack, (Hl, Wl))[:, :2, :]
        
    imstack /= im.max()
    im /= im.max()

    return im, imstack, ecc_mats

def get_imstack(im, scale, shift_max=10, theta_max=np.pi/12, nshifts=5):
    '''
        Obtain synthetically generated, low resolution images of im, with
        random shifts.
        
        Inputs:
            im: Input high resolution image
            scale: Downsampling factor (> 1)
            theta_max: Maximum angle of rotation
            nshifts: Number of shifted images to obtain
            perturb_coords: If True, perturb the coordinates to study the effect
                of erroneous registration
            
        Outputs:
            imstack: Stack of images
            coordstack: Stack of (x ,y) coordinates for each image
    '''
    H, W, _ = im.shape
    shifts = np.random.randint(-shift_max, shift_max, size=[nshifts, 2])
    thetas = (2*np.random.rand(nshifts)-1)*theta_max
    Y, X = np.mgrid[:H, :W]
    
    tmp = cv2.resize(im, None, fx=1/scale, fy=1/scale)
    Hl, Wl, _ = tmp.shape
    
    imstack = np.zeros((nshifts, Hl, Wl, 3), dtype=np.float32)
    Xstack = np.zeros((nshifts, Hl, Wl), dtype=np.float32)
    Ystack = np.zeros_like(Xstack)
    mats = np.zeros((nshifts, 2, 3))
    
    # Ensure first shift and theta are zero
    shifts[0, :] = 0
    thetas[0] = 0
    
    coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), np.ones((H*W, 1))))
    
    for idx in range(nshifts):
        shift = shifts[idx, :]
        theta = thetas[idx]
        
        mat = getEuclidianMatrix(theta, shift)
        mats[idx, ...] = mat
        coords_new = mat.dot(coords.T).T
        Xnew = coords_new[:, 0].reshape(H, W)
        Ynew = coords_new[:, 1].reshape(H, W)
        
        Xnew = cv2.resize(Xnew, (Wl, Hl), interpolation=cv2.INTER_LINEAR)
        Ynew = cv2.resize(Ynew, (Wl, Hl), interpolation=cv2.INTER_LINEAR)
        
        imstack[idx, ...] = cv2.remap(im, Xnew.astype(np.float32),
                                      Ynew.astype(np.float32),
                                      cv2.INTER_LINEAR)
        Xstack[idx, ...] = 2*Xnew/W - 1
        Ystack[idx, ...] = 2*Ynew/H - 1
        
    return imstack, Xstack, Ystack, mats
        

def get_downsampled_shifted_images(im, scale, shift_max=10, 
                                   theta_max=np.pi/12, nshifts=5,
                                   perturb_coords=False):
    '''
        Obtain synthetically generated, low resolution images of im, with
        random shifts.
        
        Inputs:
            im: Input high resolution image
            scale: Downsampling factor (> 1)
            theta_max: Maximum angle of rotation
            nshifts: Number of shifted images to obtain
            perturb_coords: If True, perturb the coordinates to study the effect
                of erroneous registration
            
        Outputs:
            imstack: Stack of images
            coordstack: Stack of (x ,y) coordinates for each image
    ''' 
    H, W = im.shape
    shifts = np.random.randint(-shift_max, shift_max, size=[nshifts, 2])
    thetas = (2*np.random.rand(nshifts)-1)*theta_max
    Y, X = np.mgrid[:H, :W]
    
    tmp = cv2.resize(im, None, fx=1/scale, fy=1/scale)
    Hl, Wl = tmp.shape
    
    imstack = np.zeros((nshifts, Hl, Wl), dtype=np.float32)
    Xstack = np.zeros_like(imstack)
    Ystack = np.zeros_like(imstack)
    
    # Ensure first shift and theta are zero
    shifts[0, :] = 0
    thetas[0] = 0
    
    for idx in range(nshifts):
        shift = shifts[idx, :]
        theta = thetas[idx]
        
        # Shift
        Xshifted = X - shift[1]
        Yshifted = Y - shift[0]
        
        # Rotate
        Xrot = (Xshifted-W/2)*np.cos(theta) - (Yshifted-H/2)*np.sin(theta) + W/2
        Yrot = (Xshifted-W/2)*np.sin(theta) + (Yshifted-H/2)*np.cos(theta) + H/2
        
        Xnew = cv2.resize(Xrot, (Wl, Hl), interpolation=cv2.INTER_AREA)
        Ynew = cv2.resize(Yrot, (Wl, Hl), interpolation=cv2.INTER_AREA)
        
        imstack[idx, ...] = cv2.remap(im, Xnew.astype(np.float32),
                                     Ynew.astype(np.float32), cv2.INTER_AREA)
        
        if perturb_coords:
            # Now ... let's generate noisy estimates
            Xshifted = X - (1 + np.random.randn(1)*1e-2)*shift[1]
            Yshifted = Y - (1 + np.random.randn(1)*1e-2)*shift[0]
            
            theta = (1 + np.random.randn(1)*1e-2)*theta
            Xrot = (Xshifted-W/2)*np.cos(theta) -\
                   (Yshifted-H/2)*np.sin(theta) + W/2
            Yrot = (Xshifted-W/2)*np.sin(theta) +\
                   (Yshifted-H/2)*np.cos(theta) + H/2
            
            Xnew = cv2.resize(Xrot, (Wl, Hl), interpolation=cv2.INTER_AREA)
            Ynew = cv2.resize(Yrot, (Wl, Hl), interpolation=cv2.INTER_AREA)
        
        Xstack[idx, ...] = 2*Xnew/W - 1
        Ystack[idx, ...] = 2*Ynew/H - 1
        
    return imstack, Xstack, Ystack, shifts, thetas

def register_stack(imstack, full_res, method=StackReg.RIGID_BODY):
    '''
        Register a stack of images and get coordinates
        
        Inputs:
            imstack: nimg x H x W stack of images
            full_res: Resolution at which images will be super resolved
            method: Method to use for registration. Default is StackReg.RIGID_BODY
            
        Outputs:
            reg_mats: (nimg, 2, 3) dimensional registration matrices
    '''
    nimg, H, W = imstack.shape
    Hr, Wr = full_res

    imstack_full = np.zeros((nimg, Hr, Wr))
    
    # Upsample the images
    for idx in range(nimg):
        imstack_full[idx, ...] = cv2.resize(imstack[idx, ...], (Wr, Hr),
                                            interpolation=cv2.INTER_AREA)
    
    # Now register the stack
    reg = StackReg(method)
    reg_mats = reg.register_stack(imstack_full, reference='first', verbose=True)
    
    return reg_mats

def invert_regstack(regstack):
    '''
        Invert affine matrices
    '''
    nimg = regstack.shape[0]
    
    regstack_inv = np.zeros_like(regstack)
    
    last_row = np.zeros((1, 3))
    last_row[0, 2] = 1
    for idx in range(nimg):
        mat = linalg.inv(np.vstack((regstack[idx, ...], last_row)))[:2, :]
        regstack_inv[idx, ...] = mat

    return regstack_inv

def mat2coords(reg_stack, full_res, low_res):
    '''
        Computed 2D coordinates from affine matrices
        
        Inputs:
            reg_stack: (nimg, 2, 3) registration stack
            res: Resolution of images
    '''
    nimg, _, _ = reg_stack.shape
    H, W = full_res
    Y, X = np.mgrid[:H, :W]
    
    Hl, Wl = low_res
    
    coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), np.ones((H*W, 1))))
    
    Xstack = np.zeros((nimg, Hl, Wl), dtype=np.float32)
    Ystack = np.zeros_like(Xstack)
    
    last_row = np.zeros((1, 3))
    last_row[0, 2] = 1
    for idx in range(nimg):
        mat = linalg.inv(np.vstack((reg_stack[idx, ...], last_row)))
        coords_new = mat.dot(coords.T).T
        Xstack[idx, ...] = cv2.resize(2*coords_new[:, 0].reshape(H, W)/W - 1,
                                      (Wl, Hl), interpolation=cv2.INTER_AREA)
        Ystack[idx, ...] = cv2.resize(2*coords_new[:, 1].reshape(H, W)/H - 1,
                                      (Wl, Hl), interpolation=cv2.INTER_AREA)
        
    return Xstack, Ystack
        
def param2theta(params, w, h):
    '''
        Convert affine matrix to parameter that torch can use
        
        Inputs:
            params: nimg x 2 x 3 affine matrices
            w, h: Width and height of the image
            
        Outputs:
            theta: Matrix to use with grid_sample (for example)
            
        Reference:
        https://discuss.pytorch.org/t/how-to-convert-an-affine-transform-matrix-into-theta-to-use-torch-nn-functional-affine-grid/24315/4
    '''
    last_row = np.zeros((1, 3), dtype=np.float32)
    last_row[0, 2] = 1
    theta = np.zeros_like(params)
    
    for idx in range(params.shape[0]):
        param = np.vstack((params[idx, ...], last_row))
        param = np.linalg.inv(param)
        theta[idx,0,0] = param[0,0]
        theta[idx,0,1] = param[0,1]*h/w
        theta[idx,0,2] = param[0,2]*2/w + theta[idx,0,0] + theta[idx,0,1] - 1
        #theta[idx, 0, 2] = param[0, 2]*2/w + param[0, 0] + param[0, 1] - 1
        theta[idx,1,0] = param[1,0]*w/h
        theta[idx,1,1] = param[1,1]
        theta[idx,1,2] = param[1,2]*2/h + theta[idx,1,0] + theta[idx,1,1] - 1
        #theta[idx, 1, 2] = param[1, 2]*2/h + param[1, 0] + param[1, 1] - 1

    return theta

def affine2rigid(mats):
    '''
        Compute rigid body transformations from affine matrices
        
        Inputs:
            mats: (nmats, 2, 3) affine matrices
        
        Outputs:
            translations: (nmats, 2) translation array
            angles: (nmats) angles array
    '''
    # Compute average angle to reduce numerical errors
    if False:
        angles = (np.arccos(mats[:, 0, 0]) -
                np.arcsin(mats[:, 0, 1]) +
                np.arcsin(mats[:, 1, 0]) +
                np.arccos(mats[:, 1, 1]))/4.0
    angles = np.arccos(mats[:, 0, 0])
    translations = mats[:, :, 2]
    
    return angles, translations

def get_transformed_coords(theta, imsize):
    '''
        Compute transformed coordinates for given affine matrices
    '''
    B = theta.shape[0]
    H, W = imsize

    return F.affine_grid(theta, (B, 1, H, W)).reshape(-1, H*W, 2)
    
def interp_lr(imref, coords, renderer):
    '''
        Compute affine transformed images from coordinates at high resolution
        
        Inputs:
            imref: (1, 1, H, W) low resolution image, upsampled
            coords: (B, H, W, 2) high resolution coordinates
            renderer: Function to downsample the images
        Outputs:
            im_lr: (B, 1, Hl, Wl) low resolution transformed images
    '''
    B = coords.shape[0]
    im_hr = F.grid_sample(torch.repeat_interleave(imref, B, 0),
                          coords, mode='bilinear', align_corners=False)
    im_lr = renderer.integrator(im_hr)
    
    return im_lr
    
def register_stack_ecc(imstack, full_res, method=cv2.MOTION_EUCLIDEAN):
    '''
        Register a stack of images and get coordinates
        
        Inputs:
            imstack: nimg x H x W stack of images
            full_res: Resolution at which images will be super resolved
            method: Method to use for ECC registration
            
        Outputs:
            Xstack: X Coordinates for registration
            Ystack: Y Coordinates for registration
            mask: (nimg, ) dimensional mask for images that were successfully
                registered
            alignment_err: (nimg, ) dimensional array of alignment error
    '''
    nimg, H, W = imstack.shape
    Hr, Wr = full_res
    
    mask = np.zeros(nimg)
    alignment_err = np.zeros(nimg)
    
    Xstack = np.zeros((nimg, H, W), dtype=np.float32)
    Ystack = np.zeros((nimg, H, W), dtype=np.float32)
    
    imref = cv2.resize(imstack[0, ...], (Wr, Hr),
                       interpolation=cv2.INTER_LINEAR)
    mask[0] = 1
    
    Y, X = np.mgrid[:Hr, :Wr]
    X = 2*X/Wr - 1
    Y = 2*Y/Hr - 1
    
    Xstack[0, ...] = cv2.resize(X, (W, H), interpolation=cv2.INTER_LINEAR)
    Ystack[0, ...] = cv2.resize(Y, (W, H), interpolation=cv2.INTER_LINEAR)
    
    if method == cv2.MOTION_HOMOGRAPHY:
        ecc_mats = np.zeros((nimg, 3, 3))
        ecc_mats[0, 2, 2] = 1
    else:
        ecc_mats = np.zeros((nimg, 2, 3))
        
    # First image is registered ... to itself
    ecc_mats[0, 0, 0] = 1
    ecc_mats[0, 1, 1] = 1
    
    for idx in tqdm.tqdm(range(1, nimg)):
        im2 = cv2.resize(imstack[idx, ...], (Wr, Hr),
                         interpolation=cv2.INTER_LINEAR)
        try:
            mat, im2_aligned, flow = ecc_flow(imref, im2, warp_mode=method)
            mask[idx] = 1
            ecc_mats[idx, :] = mat
            Xstack[idx, ...] = cv2.resize(X - flow[..., 0], (W, H),
                                          interpolation=cv2.INTER_LINEAR)
            
            Ystack[idx, ...] = cv2.resize(Y - flow[..., 1], (W, H),
                                          interpolation=cv2.INTER_LINEAR)
            
            spatial_mask = (im2_aligned != 0)
            alignment_err[idx] = abs((imref - im2_aligned)*spatial_mask).mean()
        except:
            mask[idx] = 0
            continue
    
    # Now return the coordinates
    return Xstack, Ystack, mask, ecc_mats, alignment_err

def prune_stack(imstack, ecc_mats, full_res, thres=None):
    '''
        Prune a stack of images which are not well registered.
        
        Inputs:
            imstack: nimg x H x W stack of images
            ecc_mats: nimg x 2 x 3 stack of transformation matrices
            full_res: Full resolution size
            thres: Threshold of registration error to consider when rejecting 
                images. If None, 2*median(error_array) is used
                
        Outputs:
            imstack: nimg_good x H x W stack of good images
            ecc_mats: nimg_good x 2 x 3 stack of good transformation matrices
    '''
    nimg, Hl, Wl = imstack.shape
    H, W = full_res
    
    if thres is None:
        thres = 1

    imref = cv2.resize(imstack[0, ...], (W, H), interpolation=cv2.INTER_AREA)
    imten = torch.tensor(imref).cuda()[None, None, ...]
    imstack_ten = torch.tensor(imstack).cuda()[:, None, ...]
    imten = torch.repeat_interleave(imten, int(nimg), 0)
    
    mat = torch.tensor(ecc_mats.astype(np.float32)).cuda()
    imtrans = kornia.warp_affine(imten, mat, (Hl, Wl))
    
    imdiff = abs(imtrans - imstack_ten).cpu()[:, 0, ...]
    diff_array = (imdiff/(imstack + 1e-2*imstack.max())).mean(-1).mean(-1)
    mask = diff_array < thres
    
    imstack = np.copy(imstack[mask == 1, ...], order='C')
    ecc_mats = np.copy(ecc_mats[mask == 1, ...], order='C')
    imdiff = imdiff[mask == 1, ...]
    
    return imstack, ecc_mats, mask, imdiff
        
def flow2rgb(flow):
    '''
        Convert flow to an RGB image to visualize.
        
    '''
    H, W, _ = flow.shape

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)