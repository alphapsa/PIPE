# -*- coding: utf-8 -*-
"""
Created on Fre Jan  6 03:20:41 2023

@author: Alexis Brandeker, alexis@astro.su.se

Estimate background level

"""
import numpy as np
from .pipe_statistics import sigma_clip
from .reduce import aperture

def estimate(data_frame):
    """Very simple algorithm for estimating background level: compute
    the sigma-clipped median. Assumes background stars have been 
    been subtracted/masked (by setting corresponding pixels to NaN).
    Hot pixels can also be set to NaN prior to sigma-clipping.
    """
    sel = sigma_clip(data_frame, clip=3, niter=10)
    return np.nanmean(data_frame[sel])


def cube_estimate(data_cube):
    """Single-threaded application of above background level estimate
    of frames in cube.
    """
    return np.array([estimate(frame) for frame in data_cube])


def ring_median(data_cube, inrad=0.9, clip=3, niter=10):
    """Computes the median all pixels for all frames in a
    ring covering from inrad to the outer radius. Used to 
    offset the background level for the whole cube to be
    less affected by the wings of the target PSF.
    """
    rad = inrad*0.5*np.min(data_cube[0].shape[1:])
    apt = aperture(data_cube[0].shape, radius=rad)
    data_cube[:,apt] = np.nan
    sel = sigma_clip(data_cube, clip=clip, niter=niter)
    
    return np.nanmedian(data_cube[sel])
    
