# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:32:42 2020

@author: Alexis Brandeker, alexis@astro.su.se

Multi-threaded fits of PSF libraries to data cubes. Also contains
routine for binary fit.

"""
import numpy as np
import multiprocessing as mp
from .psf import fit as psf_fit, fit_binary as psf_fit_binary

try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass


def fit(psf_list, data_cube, noise_cube, mask, xc, yc,
                fitrad=50, defrad=70, krn_scl=0.3,
                krn_rad=3, bg_fit=0, nthreads=16, non_negative=False):
    """Fit a defined PSF to a frame by constructing a matrix of
    offsets. Then produce a least-square fit of base functions
    defined by the PSF offset according to the matrix. The idea is
    to take care of motion blur. The number of offsets are 
    defined by krn_rad, and the scale by krn_scl. In total there are
    (2*krn_rad+1)**2 offsets, offset by krn_scl pixels each.
    
    Only fit if position is within fitrad from median position.
    """
    Ndata = len(data_cube)
    xm, ym = np.median(xc), np.median(yc)
    psf_cube = np.zeros(data_cube.shape)
    scale = -np.ones(Ndata)
    bg = np.zeros(Ndata)
    w = np.zeros((Ndata, len(psf_list)))

    # Don't attempt to fit positions too far off
    no_fit = (xc-xm)**2 + (yc-ym)**2 > fitrad**2

    inparam = []
    for n in range(Ndata):
        if no_fit[n]:
            continue
        if mask.ndim == 3:
            mask0 = mask[n]
        else:
            mask0 = mask
        frame = data_cube[n]
        noise = noise_cube[n]
        
        inparam.append((psf_list, frame, noise, mask0, xc[n], yc[n], 
                fitrad, defrad, krn_scl, krn_rad, bg_fit, non_negative))
    
    with mp.Pool(nthreads) as p:
        outparam = p.starmap(psf_fit, inparam)

    m = 0
    for n in range(Ndata):
        if no_fit[n]:
            continue
        psf_cube[n], bg[n], _kmat, scale[n], w[n] = outparam[m]
        m += 1
    return psf_cube, scale, bg, w


def fit_binary(psf_list0, psf_list1, data_cube, noise_cube, mask, xc0, yc0,
                          xc1, yc1, psfrad=70, fitrad=30, krn_scl=0.3,
                          krn_rad=3, nthreads=16, fix_flux2=None, non_negative=False):
    """Fit two PSFs independently to both components of a binary, 
    in a similar way as the regular multi_psf.fit function 
    described above.
    """
    # Psf.fit_binary uses about 6 threads by itself
    nthreads0 = int(max(round(nthreads/6), 1))
    Ndata = len(data_cube)
    xm0, ym0 = np.median(xc0), np.median(yc0)
    psf_cube0 = np.zeros(data_cube.shape)
    psf_cube1 = np.zeros(data_cube.shape)
    scale0 = -np.ones(Ndata)
    scale1 = -np.ones(Ndata)
    bg = np.zeros(Ndata)
    w0 = np.zeros((Ndata, len(psf_list0)))
    w1 = np.zeros((Ndata, len(psf_list1)))

    # Don't attempt to fit positions too far off
    no_fit = (xc0-xm0)**2 + (yc0-ym0)**2 > fitrad**2

    inparam = []
    for n in range(Ndata):
        if mask.ndim == 3:
            mask0 = mask[n]
        else:
            mask0 = mask
        if no_fit[n]:
            continue
        frame = data_cube[n]
        noise = noise_cube[n]
        
        inparam.append((psf_list0, psf_list1, frame, noise, mask0, xc0[n], yc0[n], xc1[n], yc1[n], 
                psfrad, fitrad, krn_scl, krn_rad, fix_flux2, non_negative))
    
    with mp.Pool(nthreads0) as p:
        outparam = p.starmap(psf_fit_binary, inparam)

    m = 0
    for n in range(Ndata):
        if no_fit[n]:
            continue
        psf_cube0[n], psf_cube1[n], bg[n], _kmat0, _kmat1, scale0[n], scale1[n], w0[n], w1[n] = outparam[m]
        m += 1
    return psf_cube0, psf_cube1, scale0, scale1, bg, w0, w1

