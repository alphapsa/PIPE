# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:04:10 2020

@author: Alexis Brandeker, alexis@astro.su.se

A set of wrappers for multi-threded centering by setting up pools
and calling methods in the cent module.
"""
import numpy as np
import multiprocessing as mp
from .cent import (
    binary_psf as cent_binary_psf, 
    binary_psf_fix as cent_binary_psf_fix,
    psf as cent_psf,
    lucy as cent_deconvolve,
    binary_lucy as cent_binary_deconvolve
)

try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass


def deconvolve(psf_spline, cube, xi, yi, radius=30, subrad=3, nthreads=16):
    """Use deconvolution to compute PSF centres of frames in cube. xi, yi
    are initial integer pixel coordinates, radius is search radius and subrad
    is radius of area for refined centre-of-flux estimate
    """
    xcoos = np.zeros(len(cube))
    ycoos = xcoos.copy()

    inparam = []
    psf_frame = make_psf_frame(psf_spline)

    for m in range(len(cube)):
        inparam.append((psf_frame, cube[m], xi, yi, radius, subrad))
        
#    print('Using {:d} threads in centering pool'.format(nthreads))
    with mp.Pool(nthreads) as p:
        outparam = p.starmap(cent_deconvolve, inparam)
                
    for m in range(len(cube)):
        xcoos[m], ycoos[m] = outparam[m]
    return xcoos, ycoos


def make_psf_frame(psf_spline, radius=23):
    """Evaluates PSF spline to produce a normalised PSF
    frame to be used by deconvolution routines
    """
    x = np.linspace(-radius, radius, 2*radius+1)
    psf_frame = psf_spline(x, x)
    psf_frame /= np.nanmax(psf_frame)
    xx, yy = np.mgrid[:(2*radius+1), :(2*radius+1)] - radius
    mask = xx**2+yy**2 <= radius**2
    psf_frame *= mask
    return psf_frame


def binary_deconvolve(psf_spline, cube, dx, dy, xi, yi, radius=25, subrad=3, nthreads=16):
    """Use deconvolution to compute PSF centres of binary components in frames of cube. xi, yi
    are integer pixel coordinates of search window, radius is search radius and subrad
    is radius of area for refined centre-of-flux estimates.
    """
    xcoos0 = np.zeros(len(cube))
    ycoos0 = xcoos0.copy()
    xcoos1 = xcoos0.copy()
    ycoos1 = xcoos0.copy()

    inparam = []
    psf_frame = make_psf_frame(psf_spline)

    for m in range(len(cube)):
        inparam.append((psf_frame, cube[m], dx[m], dy[m], xi, yi, radius, subrad))
        
    print('Using {:d} threads in centering pool'.format(nthreads))
    with mp.Pool(nthreads) as p:
        outparam = p.starmap(cent_binary_deconvolve, inparam)
                
    for m in range(len(cube)):
        res = outparam[m]
        xcoos0[m] = res[0]
        ycoos0[m] = res[1]
        xcoos1[m] = res[2]
        ycoos1[m] = res[3]
    return xcoos0, ycoos0, xcoos1, ycoos1



def psf(psf_spline, cube, noise_cube, xc, yc,
                    mask=None, radius=30, norm=1, nthreads=16):
    """Use a PSF to find the best matching centres in a cube.
    Uses noise cube to clip bad pixels.
    """
    xcoos = np.zeros(len(cube))
    ycoos = xcoos.copy()

    inparam = []

    for m in range(len(cube)):
        frame = cube[m]
        noise = noise_cube[m]
        if mask is not None:
            if mask.ndim == 3:
                mask0 = mask[m]
            else: 
                mask0 = mask
        else: 
            mask0 = mask
        inparam.append((psf_spline, frame, noise, xc[m], yc[m], 
                     mask0, radius, norm))
        
    print('Using {:d} threads in centering pool'.format(nthreads))
    with mp.Pool(nthreads) as p:
        outparam = p.starmap(cent_psf, inparam)
                
    for m in range(len(cube)):
        res = outparam[m]
        xcoos[m] = res[0] + xc[m]
        ycoos[m] = res[1] + yc[m]
    return xcoos, ycoos


def binary_psf(psf_spline, cube, noise_cube, xc0, yc0, xc1, yc1,
                    norm0=1, norm1=1, mask=None, radius=25, nthreads=16):
    """Use a PSF to find the best matching centres in a cube.
    Uses noise cube to clip bad pixels.
    """
    xcoos0 = np.zeros(len(cube))
    ycoos0 = xcoos0.copy()
    xcoos1 = xcoos0.copy()
    ycoos1 = xcoos0.copy()
    sc0 = xcoos0.copy()
    sc1 = xcoos0.copy()

    inparam = []

    for m in range(len(cube)):
        frame = cube[m]
        noise = noise_cube[m]
        if mask is not None:
            if mask.ndim == 3:
                mask0 = mask[m]
            else: 
                mask0 = mask
        else: 
            mask0 = mask
        inparam.append((psf_spline, frame, noise, xc0[m], yc0[m],
                        xc1[m], yc1[m], norm0, norm1,
                        mask0, radius))
        
    print('Using {:d} threads in centering pool'.format(nthreads))
    with mp.Pool(nthreads) as p:
        outparam = p.starmap(cent_binary_psf, inparam)
                
    for m in range(len(cube)):
        res = outparam[m]
        xcoos0[m] = res[0] + xc0[m]
        ycoos0[m] = res[1] + yc0[m]
        sc0[m] = res[2]
        xcoos1[m] = res[3] + xc1[m]
        ycoos1[m] = res[4] + yc1[m]
        sc1[m] = res[5]
    return xcoos0, ycoos0, sc0, xcoos1, ycoos1, sc1


def binary_psf_fix(psf_spline, cube, noise_cube, xc0, yc0, dx, dy,
                    norm0=1, norm1=1, mask=None, radius=25, nthreads=16):
    """Use a PSF to find the best matching centres in a cube, using
    initial guessed positions and a fixed separation (dx, dy).
    Uses noise cube to clip bad pixels.
    """
    xcoos0 = np.zeros(len(cube))
    ycoos0 = xcoos0.copy()
    xcoos1 = xcoos0.copy()
    ycoos1 = xcoos0.copy()
    sc0 = xcoos0.copy()
    sc1 = xcoos0.copy()

    inparam = []

    for m in range(len(cube)):
        frame = cube[m]
        noise = noise_cube[m]
        if mask is not None:
            if mask.ndim == 3:
                mask0 = mask[m]
            else: 
                mask0 = mask
        else: 
            mask0 = mask
        inparam.append((psf_spline, frame, noise,
                        xc0[m], yc0[m], dx[m], dy[m], norm0, norm1,
                         mask0, radius))
        
    print('Using {:d} threads in centering pool'.format(nthreads))
    with mp.Pool(nthreads) as p:
        outparam = p.starmap(cent_binary_psf_fix, inparam)
                
    for m in range(len(cube)):
        res = outparam[m]
        xcoos0[m] = res[0] + xc0[m]
        ycoos0[m] = res[1] + yc0[m]
        sc0[m] = res[2]
        xcoos1[m] = res[0] + xc0[m] + dx[m]
        ycoos1[m] = res[1] + yc0[m] + dy[m]
        sc1[m] = res[3]
    return xcoos0, ycoos0, sc0, xcoos1, ycoos1, sc1



