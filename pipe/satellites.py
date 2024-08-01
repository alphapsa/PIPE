# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:15:00 2024

@author: Alexis Brandeker, alexis@astro.su.se

Module that detects and reproduces satellite streaks in PIPE residuals.

Note that for skimage images need to be of the correct byte order:
    
    img = img.byteswap().newbyteorder()


"""
import warnings
import numpy as np
import skimage


def find_angle(img, edge=20, Ncoarse=40, Nfine=20, Niter=3):
    """Find the best angle by which to rotate the image such that
    the horisontal median gives the highest contrast. Used to find
    angle of satellite streaks. Do not apply median filter closer
    than edge to the edge (too few pixels make it noisy). Only check
    angles between rotmin and rotmax. 
    """
    ang, cont = max_contrast(img, N=Ncoarse, edge=edge)
    return refine_angle(img, angles=ang, contrast=cont,
                        Niter=Niter, edge=edge, Nres=Nfine)


def find_satellites(angles, contrasts, klip=5):
    """Use statistic on median contrasts to identify
    what frames likely contain satellite crossings, and what
    frames likely contain significantly anisotropic backgrounds
    """
    fwhm_lim = 40
    
    dev = compute_deviation(contrasts)
    base = np.median(dev)
    candidates = np.where(dev > (base + klip*_mad(dev)))[0]
    
    sat, bg = [], []
    
    for cand in candidates:
        if fwhm(angles, contrasts[cand]) < fwhm_lim:
            sat.append(cand)
        else:
            bg.append(cand)
    return sat, bg        


def make_sat_bg(img, angles, contrast, Niter=3, edge=20, Nres=20, dev2_lim=10):
    """Make background image corresponding to satellite crossing.
    Check if there is a second satellite crossing, and if so 
    corrects for that as well.
    """
    angle = refine_angle(img, angles, contrast,
                         Niter=Niter, edge=edge, Nres=Nres)
    bg = make_aniso_bg(img, angle)
    res = img - bg
    
    ang2, cont2 = max_contrast(res)
    dev = compute_deviation([cont2])[0]
    
    if dev < dev2_lim:
        return bg, 1
    bg = make_aniso_iter(img, nmax=3, niter=Niter, edge=edge)
    return bg, 2


def make_aniso_iter(img, nmax=3, niter=3, edge=20, cent=25):
    """ Find multiple anisotropic directions and iteratievely produce
    a background model
    """

    def sub_bgs(img, bgs, not_bg=None):
        res = img.copy()
        for n, bg in enumerate(bgs):
            if n is not not_bg:
                res -= bg
        return res

    bgs = []
    for n in range(niter):
        for m in range(nmax):
            res = sub_bgs(img, bgs, not_bg=m)
            angle = find_angle(res, edge=edge)
            bg = make_aniso_bg(res, angle, cent=cent)

            if m >= len(bgs):
                bgs.append(bg)
            else:
                bgs[m] = bg

    return np.sum(np.array(bgs), axis=0)
            


def compute_deviation(contrasts):
    """For each contrast array, estimate the maximum deviation
    from the base. Large deviations are indicative of significant
    anisotropies
    """
    mad = _mad(contrasts, axis=1)
    sel = mad > 0       # Do not divide by zero
    mad[sel==0] = 1
    return (np.max(contrasts, axis=1) - np.percentile(contrasts, 10, axis=1)) * sel/mad


def fwhm(x, f):
    """Compute a robust FWHM.
    """
    # Find and subtract base
    fb = f - np.percentile(f, 10)
    # Find index of max
    peak = np.max(fb)    
    x_max = np.where(fb==peak)[0][0]
    x0 = np.where(fb >= 0.5*peak)[0][0]
    if x_max < len(x)-1:
        ind = np.where(fb[x_max:] < 0.5*peak)[0]
        if len(ind) > 0:
            x1 = x_max + np.min(ind)
        else:
            x1 = -1
    else:
        x1 = -1
    return x[x1]-x[x0]


def make_aniso_bg(img, angle, cent=25):
    """Take the median of the image along lines at angle,
    then produce an image of same size but filled with the median
    stretched along lines at the same angle. Used to model linear 
    structures at angle in the image, e.g. satellite streaks.
    """
    mask = make_mask(img, cent=cent)
    rot = skimage.transform.rotate(img, angle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = np.nanmedian(mask*rot, axis=1)
    bg = np.ones_like(img)*profile[:,None]
    bg[np.isnan(bg)] = 0
    rotbg = skimage.transform.rotate(bg, -angle)
    rotbg[np.isnan(img)] = np.nan
    return rotbg


def make_mask(img, cent=25):
    """Produce central mask of radius cent to remove effects of target.
    """
    mask = np.ones_like(img)
    X,Y = np.mgrid[:img.shape[0],:img.shape[1]]
    R2 = (X-img.shape[0]/2)**2+(Y-img.shape[1]/2)**2
    mask[np.isnan(img)] = np.nan
    mask[R2 <= cent**2] = np.nan
    return mask


def max_contrast(img, N=40, edge=20):
    """Compute max median contrast along 2*N angles between -90 to 90 degrees
    Higher performance than max_contrast_hires, but computes for full 180 deg 
    range.
    """
    Ddeg = 90.0/N
    rotdegs = np.arange(0, 90, Ddeg)
    N = len(rotdegs)
    angles = np.zeros(2*N)
    angles[:N] = rotdegs-90
    angles[N:] = rotdegs
    contrast = np.zeros(2*N)
    
    for n,angle in enumerate(rotdegs):
        rot = skimage.transform.rotate(img, angle)
        rot[np.isnan(img)] = np.nan
        contrast[n] = (np.nanmax(np.nanmedian(rot[:, edge:-edge], axis=0)) - 
                         np.nanmin(np.nanmedian(rot[:, edge:-edge], axis=0)))
        contrast[n+N] = (np.nanmax(np.nanmedian(rot[edge:-edge, :], axis=1)) - 
                         np.nanmin(np.nanmedian(rot[edge:-edge, :], axis=1)))
    return angles, contrast


def max_contrast_hires(img, rotdegs, edge=20):
    """Compute max contrast along defined angles
    """
    contrast = np.zeros(len(rotdegs))
    
    for n,angle in enumerate(rotdegs):
        rot = skimage.transform.rotate(img, angle)
        rot[np.isnan(img)] = np.nan
        contrast[n] = (np.nanmax(np.nanmedian(rot[edge:-edge, :], axis=1)) - 
                   np.nanmin(np.nanmedian(rot[edge:-edge, :], axis=1)))
    return contrast


def refine_angle(img, angles, contrast, Niter=3, edge=20, Nres=20):
    """Find the best angle by which to rotate the image such that
    the horisontal median gives the highest contrast. Used to find
    angle of satellite streaks. Do not apply median filter closer
    than edge to the edge (too few pixels make it noisy). Only check
    angles between rotmin and rotmax. 
    """
    max_cont = np.max(contrast)
    max_ang = np.mean(angles[np.where(contrast==max_cont)[0]])
    ang_res = (angles[-1]-angles[0])/len(angles)
    
    if Niter > 1:
        zoom_ang = np.linspace(-1.5, 1.5, Nres)*ang_res + max_ang
        zoom_cont = max_contrast_hires(img, zoom_ang)
        return refine_angle(img, zoom_ang, zoom_cont,
                            Niter=Niter-1, edge=edge, Nres=Nres)
    return max_ang


def _mad(f, axis=0):
    return np.nanmedian(np.abs(np.diff(f, axis=axis)), axis=axis)

