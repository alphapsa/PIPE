"""
Created on Thu Mar  5 14:25:48 2020

@author: Alexis Brandeker, alexis@astro.su.se

A collection of methods to find the position of a PSF on the
detectior. Includes centre of flux and PSF fitting, also for 
a simultaneous fit of the two components in unresolved binaries.

The more computation intensive PSF centering are called from 
multi-threaded routines in the multi_cent module.
"""
import numpy as np
from scipy.optimize import minimize
import warnings


__all__ = ['flux', 'psf', 'binary_psf', 'binary_psf_fix']

def flux(cube):
    """Compute center of pixel values for cube of rectangular 
    frames. (xc, yc) is returned (in pixel coordinates of axis 1 and 2)
    """
    x = np.arange(cube.shape[1])
    y = np.arange(cube.shape[2])
    csum = np.sum(cube, (1,2))
    xcent = np.sum(cube * x[None, :, None], (1,2)) / csum
    ycent = np.sum(cube * y[None, None, :], (1,2)) / csum
    return xcent, ycent


def psf(psf_spline, frame, noise, xc, yc,
               mask = None, radius = 24, norm=1):
    """Use a PSF to find the best matching centre in a frame.
    Uses noise frame to clip bad pixels.
    """
    from .reduce import coo_mat
    c_iter = 4
    clip = 10
    xmat, ymat = coo_mat(frame.shape, xc, yc)
    xcoo =  np.arange(frame.shape[0]) - xc
    ycoo =  np.arange(frame.shape[1]) - yc
    aperture0 = xmat**2+ymat**2 <= radius**2
    if mask is not None:
        aperture0 *= mask
    
    sel = aperture0
    frame_pix = frame[sel]
    noise_pix2 = noise[sel]**2
    
    def chi(inparam):
        dx, dy, scale = inparam
        psf_pix = psf_spline(ycoo-dy, xcoo-dx)[sel]
        return np.sum((frame_pix-scale*psf_pix)**2/noise_pix2)

    init_param = np.array([0, 0, norm])
    for n in range(c_iter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(chi, init_param)
        psf_frame = psf_spline(ycoo-res.x[1], xcoo-res.x[0])
        sel = aperture0 * (np.abs(frame-res.x[2]*psf_frame) < 
                                clip*noise)
        frame_pix = frame[sel]
        noise_pix2 = noise[sel]**2
        init_param = res.x
        #print(res.x)
    return res.x


def binary_psf(psf_spline, frame, noise, xc0, yc0, xc1, yc1,
                   norm0=1, norm1=1, mask=None, radius=25):
    """Use PSF to simultaneously find the best centers of two stars in
    a frame, using initial guessed positions. init_flux_ratio is star0/star1
    Uses noise frame to clip bad pixels.
    """
    c_iter = 4
    clip = 10
    xmat0, ymat0 = coo_mat(frame.shape, xc0, yc0)
    xmat1, ymat1 = coo_mat(frame.shape, xc1, yc1)
    xcoo0 =  np.arange(frame.shape[0]) - xc0
    ycoo0 =  np.arange(frame.shape[1]) - yc0
    xcoo1 =  np.arange(frame.shape[0]) - xc1
    ycoo1 =  np.arange(frame.shape[1]) - yc1
    notsel0 = xmat0**2+ymat0**2 > radius**2
    notsel1 = xmat1**2+ymat1**2 > radius**2
    aperture0 = (notsel0*notsel1 == 0)
    if mask is not None:
        aperture0 *= mask
    sel = aperture0
    frame_pix = frame[sel]
    noise_pix = noise[sel]
    
    def chi(inparam):
        dx0, dy0, scale0, dx1, dy1, scale1 = inparam
        psf_frame = (scale0 * psf_spline(ycoo0-dy0, xcoo0-dx0) +
                     scale1 * psf_spline(ycoo1-dy1, xcoo1-dx1))
        psf_pix = psf_frame[sel]
        return np.sum(np.abs(frame_pix-psf_pix)**2/noise_pix**2)

    init_param = np.array([0, 0, norm0, 0, 0, norm1])
    for n in range(c_iter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(chi, init_param)
        psf_frame = (res.x[2] * psf_spline(ycoo0-res.x[1], xcoo0-res.x[0]) +
                     res.x[5] * psf_spline(ycoo1-res.x[4], xcoo1-res.x[3]))
        
        sel = aperture0 * (np.abs(frame - psf_frame) < clip*noise)
        frame_pix = frame[sel]
        noise_pix = noise[sel]
        init_param = res.x
        #print(res.x)
    return res.x
    

def binary_psf_fix(psf_spline, frame, noise, xc0, yc0, dx, dy,
                   norm0=1, norm1=1, mask=None, radius=25):
    """Use PSF to simultaneously find the best centers of two stars in
    a frame, using an initial guessed position and a fixed separation (relpos).
    init_flux_ratio is star0/star1. Uses noise frame to clip bad pixels.
    Ignore pixels outside of radius from each compnonent.
    """
    c_iter = 4
    clip = 10
    xmat0, ymat0 = coo_mat(frame.shape, xc0, yc0)
    xmat1, ymat1 = coo_mat(frame.shape, xc0+dx, yc0+dy)
    xcoo0 =  np.arange(frame.shape[0]) - xc0
    ycoo0 =  np.arange(frame.shape[1]) - yc0
    xcoo1 =  np.arange(frame.shape[0]) - xc0 - dx
    ycoo1 =  np.arange(frame.shape[1]) - yc0 - dy
    notsel0 = xmat0**2+ymat0**2 > radius**2
    notsel1 = xmat1**2+ymat1**2 > radius**2
    aperture0 = (notsel0*notsel1 == 0)
    if mask is not None:
        aperture0 *= mask
    
    sel = aperture0
    frame_pix = frame[sel]
    noise_pix = noise[sel]
    
    def chi(inparam):
        ddx, ddy, scale0, scale1 = inparam
        psf_frame = (scale0 * psf_spline(ycoo0-ddy, xcoo0-ddx) +
                     scale1 * psf_spline(ycoo1-ddy, xcoo1-ddx))
        psf_pix = psf_frame[sel]
        return np.sum(np.abs(frame_pix-psf_pix)**2/noise_pix**2)

    init_param = np.array([0, 0, norm0, norm1])
    for n in range(c_iter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(chi, init_param)
        psf_frame = (res.x[2] * psf_spline(ycoo0-res.x[1], xcoo0-res.x[0]) +
                     res.x[3] * psf_spline(ycoo1-res.x[1]-dy, 
                          xcoo1-res.x[0]-dx))
        
        sel = aperture0 * (np.abs(frame - psf_frame) < clip*noise)
        frame_pix = frame[sel]
        noise_pix = noise[sel]
        init_param = res.x
    return res.x
