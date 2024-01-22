# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 00:50:00 2023

@author: Alexis Brandeker, alexis@astro.su.se

Multi processing module for background stars to use the catalog
file of background stars (retrieved from Gaia) to produce
synthetic images of the field of view observed by CHEOPS, using
an empirical PSF.
"""

import numpy as np
import multiprocessing as mp
from .syntstar import make_bg_frame, refine_bg_model
from .syntstar import make_bg_psf_mask, make_bg_circ_mask

try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass


def make_star_bg(shape, psf_ids, psfs, work_cats, skip=[0], krn_scl=0.3, krn_rad=3, nthreads=1):
    in_params = []
    xkern = np.linspace(-krn_rad, krn_rad, 2*krn_rad + 1)
    xkmat, ykmat = np.meshgrid(xkern, xkern)
    selk = (xkmat**2+ykmat**2) <= krn_rad**2
    kx = krn_scl*xkmat[selk]
    ky = krn_scl*ykmat[selk]

    for wc in work_cats:
        in_params.append((shape, wc, psf_ids[:wc.catsize], psfs, skip, kx, ky))

    with mp.Pool(nthreads) as p:
        bgstars = p.starmap(make_bg_frame, in_params)

    return np.array(bgstars)


def refine_star_bg(starids, datacube, noisecube, maskcube, modelcube, psf_norm,
         work_cats, psf_ids, psfs, krn_scl=0.3, krn_rad=3, nthreads=1):
    in_params = []
    for n in range(len(work_cats)):
        in_params.append((starids, datacube[n], noisecube[n], maskcube[n],
                          modelcube[n], psf_norm, work_cats[n],
                          psf_ids[:work_cats[n].catsize],
                          psfs, krn_scl, krn_rad))

    with mp.Pool(nthreads) as p:
        refined_cats = p.starmap(refine_bg_model, in_params)

    return refined_cats


def make_bg_circ_mask_cube(shape, work_cats, skip=[0], radius=20, nthreads=1):
    in_params = []

    for wc in work_cats:
        in_params.append((shape, wc, skip, radius))

    with mp.Pool(nthreads) as p:
        mask = p.starmap(make_bg_circ_mask, in_params)

    return np.array(mask)


def make_bg_psf_mask_cube(shape, psf_ids, psfs, work_cats, skip=[0], krn_scl=0.3,
                          krn_rad=3, radius=25, level=0.1, nthreads=1):
    in_params = []
    xkern = np.linspace(-krn_rad, krn_rad, 2*krn_rad + 1)
    xkmat, ykmat = np.meshgrid(xkern, xkern)
    selk = (xkmat**2+ykmat**2) <= krn_rad**2
    kx = krn_scl*xkmat[selk]
    ky = krn_scl*ykmat[selk]

    for wc in work_cats:
        in_params.append((shape, wc, psf_ids[:wc.catsize], psfs, skip, kx, ky, radius, level))

    with mp.Pool(nthreads) as p:
        mask = p.starmap(make_bg_psf_mask, in_params)

    return np.array(mask)
