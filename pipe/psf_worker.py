# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:41:54 2020

@author: Alexis Brandeker, alexis@astro.su.se

This module is separated from make_multi_psf so that the functions here can be
called from multiple threads, as calling functions in the same module can
cause all sorts of problems in python.
"""

import numpy as np
from scipy.interpolate import LSQBivariateSpline
import time

def make_psf_spline(pixtab, radius=35, polydeg=3, niter=3):
    """least-square fit a 2D spline to the supplied table of pixel
    coordinates, values, and error on values. The spline is returned.
    """
    subrad = radius-1
    t = np.linspace(-subrad, subrad, int(2*subrad+1))
    sel = np.ones(len(pixtab),dtype='?')
    w0 = np.nanmedian(pixtab[:,3])
    if niter > 1:   # Make first iteration more robust by using first order spline
        polydeg0 = 1
    else:
        polydeg0 = polydeg
    for n in range(niter):
#        print('{:s} --- Make PSF: iter {:d}/{:d}'.format(time.asctime(), n+1, niter))
        psf_spline = LSQBivariateSpline(x=pixtab[sel,0], 
                                         y=pixtab[sel,1],
                                         z=pixtab[sel,2],
                                         w=w0/pixtab[sel,3],
                                         tx=t,ty=t,
                                         kx=polydeg0, ky=polydeg0)
        polydeg0=polydeg
        sel = filter_pix(psf_spline, pixtab)
        # print('    Masked: {:d} / {:d} pixels = {:.2f}%'.format(np.sum(sel==0),
        #       len(sel), 100*np.sum(sel==0)/len(sel)))
    return psf_spline


def filter_pix(psf_spline, pix, clip=3):
    """Filter out pixels deviating too much from fit.
    """
    y = psf_spline(pix[:,0], pix[:,1], grid=False)
    k = 0.1*np.max(y)
    s = np.std((pix[:,2]-y)/(np.abs(y)+k))
    selS = np.abs(pix[:,2]-y) < clip*s*(np.abs(y)+k)
    return selS