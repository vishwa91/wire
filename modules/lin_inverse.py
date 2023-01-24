#!/usr/bin/env python

import os
import sys
import glob
import tqdm
import pdb

import numpy as np
from scipy import signal

import torch
from torch import nn
import kornia

import matplotlib.pyplot as plt
import cv2

def radon(imten, angles, is_3d=False):
    '''
        Compute forward radon operation
        
        Inputs:
            imten: (1, nimg, H, W) image tensor
            angles: (nangles) angles tensor -- should be on same device as 
                imten
        Outputs:
            sinogram: (nimg, nangles, W) sinogram
    '''
    nangles = len(angles)
    imten_rep = torch.repeat_interleave(imten, nangles, 0)
    
    imten_rot = kornia.geometry.rotate(imten_rep, angles)
    
    if is_3d:
        sinogram = imten_rot.sum(2).squeeze().permute(1, 0, 2)
    else:
        sinogram = imten_rot.sum(2).squeeze()
        
    return sinogram

def get_video_coding_frames(video_size, nframes):
    '''
        Get masks for video CS
        
        Inputs:
            video size: Size of the video cube
            nframes: Number of frames to combine into a single frame
            
        Outputs:
            masks: Binary masks of the same size as video_size            
    '''
    H, W, totalframes = video_size

    X, Y = np.mgrid[:H, :W]
    
    indices = np.random.randint(0, nframes, (H, W))
    masks_sub = np.zeros((H, W, nframes))
    masks_sub[X, Y, indices] = 1
    
    masks = np.tile(masks_sub, [1, 1, totalframes//nframes + 1])
    
    return masks[..., :totalframes]   

def video2codedvideo(video_ten, masks_ten, nframes):
    '''
        Convert video to coded video, similar to Hitomi et al.
        
        Inputs:
            video_ten: (1, totalframes, H, W) video tensor
            masks_ten: (1, totalframes, H, W) mask tensor
            nframes: Number of frames to combine together 
            
        Outputs:
            codedvideo_ten: (1, totalframems//nframes + 1, H, W) coded video
    '''     
    codedvideo_list = []
    
    for idx in range(0, video_ten.shape[1], nframes):
        video_chunk = video_ten[:, idx:idx+nframes, :, :]
        masks_chunk = masks_ten[:, idx:idx+nframes, :, :]
        
        codedvideo = (video_chunk*masks_chunk).sum(1, keepdim=True)
        codedvideo_list.append(codedvideo)    
    
    if idx < video_ten.shape[1]:
        video_chunk = video_ten[:, idx:, :, :]
        masks_chunk = masks_ten[:, idx:, :, :]
        
        codedvideo = (video_chunk*masks_chunk).sum(1, keepdim=True)
        codedvideo_list.append(codedvideo)    
        
    codedvideo_ten = torch.cat(codedvideo_list, dim=1)
    
    return codedvideo_ten