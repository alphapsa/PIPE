# -*- coding: utf-8 -*-
"""
Created on Fre Jan  6 03:20:41 2023

@author: Alexis Brandeker, alexis@astro.su.se

Estimate background level, multi-threaded version

"""
import numpy as np
import multiprocessing as mp
from .level import estimate as level_estimate

try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass


def estimate(data_cube, nthreads=16):
    """Estimate background levels of scattered light for datacube. Assumes 
    background stars have been subtracted or masked. Returns array of estimated
    levels.
    """
    inparam = [(frame,) for frame in data_cube]
        
    with mp.Pool(nthreads) as p:
        levels = p.starmap(level_estimate, inparam)
                
    return np.array(levels)

