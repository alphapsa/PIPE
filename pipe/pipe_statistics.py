# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:00:44 2020

@author: Alexis Brandeker, alexis@astro.su.se

A collection of routines and tools that analyse both extracted
and non-extracted data in various ways, including rudimentary aperture
photometry.

"""


import numpy as np

def mad(series):
    """Compute normalised mean absolute deviation (MAD) for an
    array, return in ppm
    """
    return np.nanmedian(np.abs(np.diff(series/np.nanmedian(series))))*1e6


def sigma_clip(data, clip=5, niter=5):
    """A simple 1D sigma-clipping routine that returns
    a boolean array of indices to non-clipped data    
    """
    ind = np.ones(data.shape, dtype='?')
    for _n in range(niter):
        sigma = np.nanstd(data[ind])
        m = np.nanmedian(data[ind])
        ind = np.abs(data-m) <= clip*sigma
    return ind

