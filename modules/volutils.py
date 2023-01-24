#!/usr/bin/env python

import sys
import importlib
import time
import itertools
import os
import pdb
import copy

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
from scipy.interpolate import RegularGridInterpolator as rgi

import cv2
import torch
import open3d as o3d
import mcubes

import matplotlib.pyplot as plt
plt.gray()

def add_noise(volume, max_shift):
    '''
        Uniformly jitter the values at each coordinate
        
        Inputs:
            volume: HxWxT binary volume 
            max_shift: Maximum allowable jitter at each pixel
            
        Outputs:
            volume_noisy: Noisy volume
    '''
    batch_size = int(50e7)
    H, W, T = volume.shape
    
    x = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    z = np.linspace(-1, 1, T)
    
    X, Y, Z = np.meshgrid(x, y, z)
    
    Xn = np.clip(X + (2*np.random.rand(H, W, T) - 1)*max_shift/H, -1, 1)
    Yn = np.clip(Y + (2*np.random.rand(H, W, T) - 1)*max_shift/W, -1, 1)
    Zn = np.clip(Z + (2*np.random.rand(H, W, T) - 1)*max_shift/T, -1, 1)
    
    func = rgi((x, y, z), volume, method='nearest')
        
    coords = np.hstack((Xn.reshape(-1, 1), Yn.reshape(-1, 1), Zn.reshape(-1, 1)))
    volume_noisy = np.zeros(H*W*T, dtype=np.float32)
    for idx in range(0, coords.shape[0], batch_size):
        idx2 = min(idx+batch_size, H*W*T-1)
        volume_noisy[idx:idx2] = func(coords[idx:idx2, :])
    
    volume_noisy = np.transpose(volume_noisy.reshape(H, W, T), [1, 0, 2])
    volume_noisy[volume_noisy <= 0.5] = 0
    volume_noisy[volume_noisy > 0.5] = 1
    
    return volume_noisy

def get_IoU_batch(preds, gt, thres=None, maxpoints=pow(2, 24)):
    intersection_list = []
    union_list = []
    
    preds = preds.flatten()
    gt = gt.flatten()

    for b_idx in range(0, preds.numel(), maxpoints):
        b_idx2 = min(preds.numel(), b_idx+maxpoints)
        intersection, union = get_I_and_U(preds[b_idx:b_idx2],
                                          gt[b_idx:b_idx2],
                                          thres)
        
        intersection_list.append(intersection)
        union_list.append(union)
        
    return sum(intersection_list)/sum(union_list)

def get_IoU(preds, gt, thres=None):
    intersection, union = get_I_and_U(preds, gt, thres)
    return intersection/union
    
def get_I_and_U(preds, gt, thres=None):
    if thres is not None:
        preds[preds < thres] = 0.0
        preds[preds >= thres] = 1.0
        
    if type(preds) == np.ndarray:
        intersection = np.logical_and(preds, gt).sum()
        union =  np.logical_or(preds, gt).sum()
    else:
        intersection =  torch.logical_and(preds.cuda(), gt.cuda()).sum()
        union = torch.logical_or(preds.cuda(), gt.cuda()).sum()
        
    return intersection, union

def export_mesh(coords, cube_res, model, batchsize, savename,
                occupancy=None, thres=0.005):
    '''
        Export 3D mesh (dae) using output from an implicit model
        
        Inputs:
            coords: (npts, 3) coordinates
            cube_res: size of cube along each dimension
            model: Implicit model that takes 3 coordinates as input and outputs
                occupancy
            batchsize: Maximum number of points to probe in each iteration
            savename: Filename to use for saving the output 3D mesh
            occupancy: (n, n, n) numpy array representing occupancy. If None,
                a new volume is generated
            thres: Threshold of marching cubes iso surface
            
        Outputs:
            occupancy: 3D occupancy volume
    '''
    if occupancy is None:
        occupancy = np.zeros((cube_res**3, 1), dtype=np.float32)
    else:
        occupancy[...] = 0
        occupancy = occupancy.reshape(-1, 1)
    
    # Compute occupancy values    
    with torch.no_grad():
        for b_idx in range(0, cube_res**3, batchsize):
            b_idx2 = min(b_idx+batchsize, cube_res**3)
            coords_sub = coords[b_idx:b_idx2, :].cuda()
            
            occupancy_sub = torch.sigmoid(model(coords_sub)).cpu().numpy()
            occupancy[b_idx:b_idx2, :] = occupancy_sub

    occupancy = occupancy.reshape(cube_res, cube_res, cube_res)
    
    # Compute vertices and faces using marching cube algorithm
    vertices, faces = mcubes.marching_cubes(occupancy, thres)
    
    # Save the file
    mcubes.export_mesh(vertices, faces, savename)
    
    # Done
    return occupancy

def get_query_coords(minlim, maxlim, cube_res, depth):
    '''
        Get regular coordinates for querying the block implicit representation
    '''
    x = np.linspace(minlim[0], maxlim[0], cube_res)
    y = np.linspace(minlim[1], maxlim[1], cube_res)
    z = np.linspace(minlim[2], maxlim[2], cube_res)
    
    X, Y, Z = np.meshgrid(x, y, z)
    coords_gen = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))
    query_dataset = BlockPCDDataset(coords_gen, None, depth)
    query_chunks, _ = query_dataset.fold()
    
    return query_chunks, query_dataset

class BlockPCDDataset(object):
    '''
        Class for representing Chunked data of a 3D point cloud
        
        Inputs:
            xyz: (npts, 3) array of points on surface
            depth: Depth of division. Each axis is divided into 2^depth blocks
            
        Outputs:
            None        
    '''
    def __init__(self, xyz, sdf=None, depth=4):
        self.xyz = xyz
        self.depth = depth
        self.cur_depth = depth
        
        if sdf is None:
            self.sdf = np.ones((xyz.shape[0], 1), dtype=np.float32)
        else:
            self.sdf = sdf
        
        self.minvals = self.xyz.min(0)[:3]
        self.maxvals = self.xyz.max(0)[:3] + 1e-6
        
        self.chunks, self.sdf, self.minlim, self.maxlim, self.indices =\
            self.build()
        
        # Prune out useless blocks
        self.clean_chunks = self.prune(self.chunks)
        self.clean_sdf = self.prune(self.sdf)

    def build(self, xyz=None, depth=None):
        '''
            Build chunks as a flat dictionary
        '''
        minvals, maxvals = self.minvals, self.maxvals
        
        if xyz is None:
            xyz = self.xyz
            sdf_vec = self.sdf
        if depth is None:
            depth = self.depth
        
        nblocks = pow(2, depth)
        
        # Rapid block idx computation
        delta = (maxvals - minvals)/nblocks
        block_idx = np.floor((xyz[:, :3] - minvals)/delta)
        block_idx = block_idx.astype(int).tolist()
        
        # Create all indices
        X, Y, Z = np.mgrid[:nblocks, :nblocks, :nblocks]
        all_idx = np.hstack((X.reshape(-1, 1),
                             Y.reshape(-1, 1),
                             Z.reshape(-1, 1)))
        # Create sampling limits
        minlim = all_idx*delta
        maxlim = (all_idx+1)*delta
                
        # Create dictionary
        chunks = {tuple(all_idx[idx, :]):[] for idx in range(nblocks**3)}
        sdf = {tuple(all_idx[idx, :]):[] for idx in range(nblocks**3)}
        indices = {tuple(all_idx[idx, :]):idx for idx in range(nblocks**3)}
                
        for idx in range(xyz.shape[0]):
            key = tuple(block_idx[idx])
            chunks[key].append(xyz[idx, :])
            sdf[key].append(sdf_vec[idx, 0])
                
        return chunks, sdf, minlim, maxlim, indices
    
    def prune(self, chunks):
        return {key:item for key, item in chunks.items() if len(item) > 0}
                
    def maxpoints(self, chunks=None):
        if chunks is None:
            chunks = self.chunks

        return max([len(item) for _, item in chunks.items()])
    
    def minpoints(self, chunks=None):
        if chunks is None:
            chunks = self.chunks

        return min([len(item) for _, item in chunks.items()])
    
    def flatten(self, chunks=None):
        if chunks is None:
            chunks = self.chunks

        chunks_list = [item for _, item in chunks.items()]
        
        chunks_sub = []
        
        for chunk in chunks_list:
            chunks_sub += chunk

        xyz_flat = np.vstack(chunks_sub)
        
        return xyz_flat
    
    def fill(self, chunks=None, sdf=None):
        if chunks is None:
            chunks = self.chunks
            sdf = self.sdf

        retchunks = copy.deepcopy(chunks)
        retsdf = copy.deepcopy(sdf)
        maxpts = self.maxpoints()
        
        for key in chunks.keys():
            curpts = len(chunks[key])
            addpts = maxpts - curpts
                        
            minlim = self.minlim[self.indices[key], :]
            maxlim = self.maxlim[self.indices[key], :]
            newpts = [minlim + (maxlim - minlim)*np.random.rand(1, 3) \
                for _ in range(addpts)]
            newsdf = [0 for _ in range(addpts)]
            
            retchunks[key] += newpts
            retsdf[key] += newsdf
            
        return retchunks, retsdf
    
    def refill(self, folded_chunks, folded_sdfs, minlim, maxlim):
        '''
            Refill coordinates where SDF values are zero
        '''
        delta = maxlim - minlim
        newpts = torch.rand_like(folded_chunks, device='cuda')
        newpts = minlim[:, None, :] + delta[:, None, :]*newpts
        folded_chunks = folded_sdfs*folded_chunks + (1 - folded_sdfs)*newpts
        
        return folded_chunks
    
    def fold(self, chunks=None, sdfs=None):
        '''
            Convert the dictionary of chunks to pytorch tensor of size,
            (nchunks, npoints, 3)
            
            Inputs:
                chunks: Dictionary of chunks
                sdf: Dictionary of SDFs
                
            NOTE: You have to call fill() before folding to ensure that all
                blocks have same number of points
        '''
        if chunks is None:
            chunks = self.chunks
            sdfs = self.sdf

        nchunks = len(chunks)
        npoints = self.maxpoints(chunks)
        
        folded_chunks = np.zeros((nchunks, npoints, 3), dtype=np.float32)
        folded_sdf = np.zeros((nchunks, npoints, 1), dtype=np.float64)
        
        for key in chunks.keys():
            cnt = self.indices[key]
            folded_chunks[cnt, :, :] = torch.tensor(np.vstack(chunks[key]))
            folded_sdf[cnt, :, 0] = torch.tensor(sdfs[key])
            
        return folded_chunks, folded_sdf
    
    def unfold(self, sdfs, sdfcube=None, cubesize=None):
        '''
            Convert a chunked list of SDFs to a cube of SDFs. Currently only
            cube with equal dimensions is supported
        '''
        nblocks = pow(2, self.depth)
        
        if sdfcube is None:
            sdfcube = np.zeros((cubesize, cubesize, cubesize), dtype=np.float32)
        else:
            cubesize = sdfcube.shape[0]
            
        blocksize = cubesize // nblocks
            
        for key in self.indices:
            x1, y1, z1 = list(key)
            x1, y1, z1 = x1*blocksize, y1*blocksize, z1*blocksize
            x2, y2, z2 = x1 + blocksize, y1 + blocksize, z1 + blocksize
            
            block  = sdfs[self.indices[key], :, 0].reshape(blocksize,
                                                           blocksize,
                                                           blocksize)
            # Not sure why, requires double transpose
            sdfcube[x1:x2, y1:y2, z1:z2] = np.transpose(block, [1, 0, 2])
            
        # Not sure why, but we have to transpose
        return np.transpose(sdfcube, [1, 0, 2])
        
    def downsample(self, chunks=None, ndepths=1, targetdepth=None):
        if chunks is None:
            chunks = self.chunks
            
        if ndepths == 0:
            return self.chunks, self.sdf, self.minlim, self.maxlim, self.indices
            
        for idx in range(ndepths):
            # Remove useless chunks
            chunks = self.prune(chunks)
            
            # Average
            chunks_ds = {key:[np.vstack(item).mean(0)] \
                            for key, item in chunks.items()}
            # Collate
            chunks_flat = self.flatten(chunks_ds)
            
            # Subdivide
            chunks, sdf, minlim, maxlim, indices = self.build(chunks_flat,
                                                        self.depth - idx - 1)
            
        # If target depth is other than None, redivide
        if targetdepth is not None:
            chunks, sdf, minlim, maxlim, indices = self.build(chunks_flat,
                                                              targetdepth)
            
        return chunks, sdf, minlim, maxlim, indices
    
def get_occupancy_cube(cube_res, sidelength, pred_occupancy, model_input,
                       display_occupancy=None):

    # get voxel idx for each coordinate
    coords = model_input['fine_abs_coords']
    if type(coords) == torch.Tensor:
        coords = coords.cpu().numpy()
    voxel_idx = np.floor((coords + 1.)/2.*(sidelength)).astype(np.int32)
    voxel_idx = voxel_idx.reshape(-1, 3)

    # init a new occupancy volume
    if display_occupancy is None:
        display_occupancy = -1 * np.ones((cube_res, cube_res, cube_res),
                                        dtype=np.float32)
    else:
        display_occupancy[...] = -1

    # assign predicted voxel occupancy values into the array
    pred_occupancy = pred_occupancy.reshape(-1, 1).detach().cpu().numpy()
    display_occupancy[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]] = \
        pred_occupancy[..., 0]
        
    return display_occupancy

def march_and_save(occupancy, mcubes_thres, savename, smoothen=False):
    '''
        Convert volumetric occupancy cube to a 3D mesh
        
        Inputs:
            occupancy: (H, W, T) occupancy volume with values going from 0 to 1
            mcubes_thres: Threshold for marching cubes algorithm
            savename: DAE file name to save
            smoothen: If True, the mesh is binarized, smoothened, and then the
                marching cubes is applied
        Outputs:
            None
    '''
    if smoothen:
        occupancy = occupancy.copy()
        occupancy[occupancy < mcubes_thres] = 0.0
        occupancy[occupancy >= mcubes_thres] = 1.0
        
        occupancy = mcubes.smooth(occupancy, method='gaussian', sigma=1)
        mcubes_thres = 0
        
    vertices, faces = mcubes.marching_cubes(occupancy, mcubes_thres)
    
    #vertices /= occupancy.shape[0]
        
    mcubes.export_mesh(vertices, faces, savename)
    
def cuboid_data(o, size=(1,1,1)):
    # https://stackoverflow.com/questions/49277753/python-matplotlib-plotting-cuboids
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    
    eps = 0.01
    l, w, l = l - eps, w - eps, h - eps
    
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos=(0,0,0), size=(1,1,1), color='b',
               edgecolor='k', alpha=1.0, ax=None,**kwargs):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos, size )
        ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1,
                        alpha=alpha, edgecolors=edgecolor, linewidth=0.1,
                        **kwargs)
