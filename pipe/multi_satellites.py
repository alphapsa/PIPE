# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:15:00 2024

@author: Alexis Brandeker, alexis@astro.su.se

Module that detects and reproduces satellite streaks in PIPE residuals.

"""

import numpy as np
import multiprocessing as mp
from .satellites import max_contrast, find_satellites, make_sat_bg, make_aniso_iter

try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass


def comp_contrasts(res_cube, nthreads=16, N=40, edge=20):
    """res_cube is cube of residuals (data-model) as found by PIPE.
    Compute max median contrast along 2*N angles between -90 to 90 degrees
    Higher performance than max_contrast_hires, but computes for full 180 deg 
    range. 
    """
    
    inparam = [(frame, N, edge) for frame in res_cube]
    
    with mp.Pool(nthreads) as p:
        outparam = p.starmap(max_contrast, inparam)
    
    angles = outparam[0][0]
    contrasts = np.zeros((len(outparam), len(outparam[0][1])))
    for n in range(len(outparam)):
        contrasts[n] = outparam[n][1]
    
    return angles, contrasts


def make_aniso_bg(res_cube, edge=20, klip=3, klip2=10, nthreads=16):
    """Find frames with satellite crossings and other background
    anisotropies, and produce a model cube of the backgrounds.
    Returns background model cube and flag array, where
    0 is no BG anomaly, 1 is satellite streak, 2 is 2 satellites,
    and 3 is complex background
    """
    
    # Compute median contrasts along a range of angels for each image
    angs, conts = comp_contrasts(res_cube, nthreads=nthreads, edge=edge)    

    # Find indices of images with likely satellite crossings and bg aniso
    sat_list, bg_list = find_satellites(angs, conts, klip=klip)
    
    # Produce synthetic BGs for relevant images
    sat_res = res_cube[sat_list]
    bg_res = res_cube[bg_list]
    Niter = 3
    Nres = 20
    
    bgs = np.zeros_like(res_cube)
    flags = np.zeros(len(bgs))

    # Model satellite streaks    
    inparam = [(frame, angs, cont, Niter, edge, Nres, klip2) 
               for frame, cont in zip(sat_res, conts[sat_list])]
    with mp.Pool(nthreads) as p:
        outparam = p.starmap(make_sat_bg, inparam)

    for n, (bg, flag) in zip(sat_list, outparam):
        bgs[n], flags[n] = bg, flag
        
    nmax = 3
    # Model anisotorpic backgrounds
    inparam = [(frame, nmax, Niter, edge) for frame in bg_res]
    with mp.Pool(nthreads) as p:
        outparam = p.starmap(make_aniso_iter, inparam)
    
    bgs[bg_list] = outparam
    flags[bg_list] = 3

    return bgs, flags    
    