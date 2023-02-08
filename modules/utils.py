#!/usr/bin/env python

'''
    Miscellaneous utilities that are extremely helpful but cannot be clubbed
    into other modules.
'''

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.ndimage as ndim
from scipy import io
from scipy.sparse.linalg import svds
from scipy import signal

import torch

# Plotting
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def rsnr(x, xhat):
    '''
        Compute reconstruction SNR for a given signal and its reconstruction.

        Inputs:
            x: Ground truth signal (ndarray)
            xhat: Approximation of x

        Outputs:
            rsnr_val: RSNR = 20log10(||x||/||x-xhat||)
    '''
    xn = lin.norm(x.reshape(-1))
    en = lin.norm((x-xhat).reshape(-1))
    rsnr_val = 20*np.log10(xn/en)

    return rsnr_val

def psnr(x, xhat):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = np.mean(pow(err, 2))

    snrval = 10*np.log10(np.max(x)/denom)

    return snrval

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas

def build_montage(images):
    '''
        Build a montage out of images
    '''
    nimg, H, W = images.shape
    
    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))
    
    montage_im = np.zeros((H*nrows, W*ncols), dtype=np.float32)
    
    cnt = 0
    for r in range(nrows):
        for c in range(ncols):
            h1 = r*H
            h2 = (r+1)*H
            w1 = c*W
            w2 = (c+1)*W

            if cnt == nimg:
                break

            montage_im[h1:h2, w1:w2] = normalize(images[cnt, ...], True)
            cnt += 1
    
    return montage_im
  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_coords(H, W, T=None):
    '''
        Get 2D/3D coordinates
    '''
    if T is None:
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    else:
        X, Y, Z = np.meshgrid(np.linspace(-1, 1, W),
                              np.linspace(-1, 1, H),
                              np.linspace(-1, 1, T))
        coords = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))
    
    return torch.tensor(coords.astype(np.float32))

def resize(cube, scale):
    '''
        Resize a multi-channel image
        
        Inputs:
            cube: (H, W, nchan) image stack
            scale: Scaling 
    '''
    H, W, nchan = cube.shape
    
    im0_lr = cv2.resize(cube[..., 0], None, fx=scale, fy=scale)
    Hl, Wl = im0_lr.shape
    
    cube_lr = np.zeros((Hl, Wl, nchan), dtype=cube.dtype)
    
    for idx in range(nchan):
        cube_lr[..., idx] = cv2.resize(cube[..., idx], None,
                                       fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
    return cube_lr

def get_inpainting_mask(imsize, mask_type='random2d', mask_frac=0.5):
    '''
        Get a 2D mask for image inpainting
        
        Inputs:
            imsize: Image size
            mask_type: one of 'random2d', 'random1d'
            mask_frac: Fraction of non-zeros in the mask
            
        Outputs:
            mask: A 2D mask image
    '''
    H, W = imsize

    if mask_type == 'random2d':
        mask = np.random.rand(H, W) < mask_frac
    elif mask_type == 'random1d':
        mask_row = np.random.rand(1, W) < mask_frac
        mask = np.ones((H, 1)).dot(mask_row)
    elif mask_type == 'bayer':
        mask = np.zeros((H, W))
        mask[::2, ::2] = 1
        
    return mask.astype(np.float32)

@torch.no_grad()
def get_layer_outputs(model, coords, imsize,
                      nfilters_vis=16,
                      get_imag=False):
    '''
        get activation images after each layer
        
        Inputs:
            model: INR model
            coords: 2D coordinates
            imsize: Size of the image
            nfilters_vis: Number of filters to visualize
            get_imag: If True, get imaginary component of the outputs
            
        Outputs:
            atoms_montages: A list of 2d grid of outputs
    '''
    H, W = imsize

    if model.pos_encode:
        coords = model.positional_encoding(coords)
        
    atom_montages = []
    
    for idx in range(len(model.net)-1):
        layer_output = model.net[idx](coords)
        layer_images = layer_output.reshape(1, H, W, -1)[0]
        
        if nfilters_vis is not 'all':
            layer_images = layer_images[..., :nfilters_vis]
        
        if get_imag:
            atoms = layer_images.detach().cpu().numpy().imag
        else:
            atoms = layer_images.detach().cpu().numpy().real
            
        atoms_min = atoms.min(0, keepdims=True).min(1, keepdims=True)
        atoms_max = atoms.max(0, keepdims=True).max(1, keepdims=True)
        
        signs = (abs(atoms_min) > abs(atoms_max))
        atoms = (1 - 2*signs)*atoms
        
        # Arrange them by variance
        atoms_std = atoms.std((0,1))
        std_indices = np.argsort(atoms_std)
        
        atoms = atoms[..., std_indices]
        
        atoms_min = atoms.min(0, keepdims=True).min(1, keepdims=True)
        atoms_max = atoms.max(0, keepdims=True).max(1, keepdims=True)
        
        atoms = (atoms - atoms_min)/np.maximum(1e-14, atoms_max - atoms_min)
        
        atoms[:, [0, -1], :] = 1
        atoms[[0, -1], :, :] = 1
        
        atoms_montage = build_montage(np.transpose(atoms, [2, 0, 1]))
        
        atom_montages.append(atoms_montage)
        coords = layer_output
        
    return atom_montages
