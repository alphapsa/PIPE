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
from skimage import restoration
import warnings


__all__ = ['flux', 'lucy', 'psf', 'binary_lucy', 'binary_psf', 'binary_psf_fix']


def flux(cube):
    """Compute center of pixel values for cube of rectangular 
    frames. (xc, yc) is returned (in pixel coordinates of axis 2 and 1)
    """
    x = np.arange(cube.shape[2])
    y = np.arange(cube.shape[1])
    csum = np.sum(cube, (1,2))
    xcent = np.sum(cube * x[None, None, :], (1,2)) / csum
    ycent = np.sum(cube * y[None, :, None], (1,2)) / csum
    return xcent, ycent


def max_frame(frame, xi, yi, rad):
    """Finds centre as brightest pixel in subimage defined by
    integer pixel coordinates xi,yi and radius rad
    """
    subim = frame[(yi-rad):(yi+rad+1), (xi-rad):(xi+rad+1)].copy()
    xx,yy = np.mgrid[-rad:(rad+1),-rad:(rad+1)]
    sel = xx**2+yy**2 >= rad**2
    subim[sel] = None
    yind, xind = np.where(subim == np.nanmax(subim))
    xcent = xind[0] + xi - rad
    ycent = yind[0] + yi - rad
    return xcent, ycent


def flux_frame(frame, xi, yi, rad):
    """Computes center of flux for circular aperture centered on
    integer pixel coordinates xi,yi and radius rad. NaNs are ignored.
    """
    subim = frame[(yi-rad):(yi+rad+1), (xi-rad):(xi+rad+1)].copy()
    subim *= (subim>0) # No negative values as they offset fit
    xx,yy = np.mgrid[-rad:(rad+1),-rad:(rad+1)]
    sel = xx**2+yy**2 >= rad**2
    subim[sel] = None
    x = np.arange(2*rad+1)
    imsum = np.nansum(subim)
    xcent = np.nansum(subim * x[None, :]) / imsum + xi - rad
    ycent = np.nansum(subim * x[:, None]) / imsum + yi - rad
    return xcent, ycent


def lucy(psf_frame, frame, xi, yi, rad=15, subrad=3, niter=100):
    """Computes center of flux of the deconvolved frame. Particularly useful for
    crowded fields. psf_frame is a PSF model evauilated on a grid of the same scale
    (per pixel) as the frame, but that can be smaller. xc, yc are initial guess
    coordinates and rad is the search radius around guess. subrad is the refined
    radius used for centre-of-flux computation. Returns estimated x, y coordinates.
    """
    image = frame/np.nanmax(frame) # Normalisation required for deconvolution
    image *= image > 0
    deconv = restoration.richardson_lucy(image, psf_frame, niter, clip=False)
    if np.isnan(deconv).all():
        return np.nan, np.nan
    xm, ym = max_frame(deconv, xi, yi, rad=rad)
    xc, yc = flux_frame(deconv, xm, ym, rad=subrad)
    return xc, yc


def psf(psf_spline, frame, noise, xc, yc,
               mask=None, radius=30, norm=1):
    """Use a PSF to find the best matching centre in a frame.
    Uses noise frame to clip bad pixels.
    """
    from .reduce import aperture
    c_iter = 4
    clip = 10
    xcoo =  np.arange(frame.shape[1]) - xc
    ycoo =  np.arange(frame.shape[0]) - yc
    aperture0 = aperture(frame.shape, radius=radius, xc=xc, yc=yc)
    if mask is not None:
        aperture0 *= mask
    
    sel = aperture0
    frame_pix = (frame[sel]/norm)
    noise_pix = (noise[sel]/norm)
    
    def chi(inparam):
        dx, dy, scale = inparam
        psf_pix = scale*psf_spline(xcoo-dx, ycoo-dy)[sel]
        return np.sum(np.abs((frame_pix-psf_pix)/noise_pix))

    init_param = np.array([0, 0, 1])
    for _n in range(c_iter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(chi, init_param)
        psf_frame = res.x[2]*psf_spline(xcoo-res.x[0], ycoo-res.x[1])
        sel = aperture0 * (np.abs(frame - psf_frame) < clip*noise)
        frame_pix = (frame[sel]/norm)
        noise_pix = (noise[sel]/norm)
        init_param = res.x
    return res.x[0:2]

def first_coo_is_primary(xc0, yc0, xc1, yc1, dx, dy):
    """For two coordinate pairs (xc0, yc0) and (xc1, yc1), checks if the
    expected displacement (dx, dy) is more consistent with first coordinate
    being the main target (primary component of binary as listed in StarCatalog)
    and the second the secondary (return True), or the reverese (return False).
    This is a robust way of identifiying the detected components of a binary 
    system, instead of relying on e.g. flux or assumed position on detector.
    """
    dxc = xc1-xc0
    dyc = yc1-yc0
    if (dxc-dx)**2+(dyc-dy)**2 < (dxc+dx)**2+(dyc+dy)**2:
        return True
    return False

def disc_index(frame, xi, yi, rad=3):
    """Produce selection array for disc in frame, centered at integer
    pixel coordinates xi, yi with radius rad.
    """
    xx,yy = np.mgrid[(-yi):(frame.shape[0]-yi), (-xi):(frame.shape[1]-xi)]
    index = xx**2 + yy**2 <= rad**2
    return index


def binary_deconvolve(psf_frame, frame, dx, dy, xi, yi, rad=30, subrad=3):
    """Computes center of flux for the two components of a binary, where the
    components are contained in an area of radius rad centerd on integer pixel
    coordinates xi, yi. subrad is the refined radius used for centre-of-flux
    for each component.  Returns estimated x, y coordinates for two components,
    where the first is the primary according to their predicted relative positions
    dx, dy.
    """
    image = frame/np.nanmax(frame) # Normalisation required for deconvolution
    image *= image > 0
    deconv, _  = restoration.unsupervised_wiener(image, psf_frame)
    xm, ym = max_frame(deconv, xi, yi, rad=rad)
    xc0, yc0 = flux_frame(deconv, xm, ym, rad=subrad)
    deconv[disc_index(deconv, xm, ym, subrad)] = None
    xm, ym = max_frame(deconv, xi, yi, rad=rad)
    xc1, yc1 = flux_frame(deconv, xm, ym, rad=subrad)

    if first_coo_is_primary(xc0, yc0, xc1, yc1, dx, dy):
        return xc0, yc0, xc1, yc1
    
    return xc1, yc1, xc0, yc0


def binary_lucy(psf_frame, frame, dx, dy, xi, yi, rad=30, subrad=3, niter=100):
    """Computes center of flux for the two components of a binary, where the
    components are contained in an area of radius rad centerd on integer pixel
    coordinates xi, yi. subrad is the refined radius used for centre-of-flux
    for each component.  Returns estimated x, y coordinates for two components,
    where the first is the primary according to their predicted relative positions
    dx, dy.
    """
    image = frame/np.nanmax(frame) # Normalisation required for deconvolution
    image *= image > 0
    deconv = restoration.richardson_lucy(image, psf_frame, niter, clip=False)
    if np.isnan(deconv).all():
        return np.nan, np.nan, np.nan, np.nan
    xm, ym = max_frame(deconv, xi, yi, rad=rad)
    xc0, yc0 = flux_frame(deconv, xm, ym, rad=subrad)
    deconv[disc_index(deconv, xm, ym, subrad)] = np.nan
    if np.isnan(deconv).all():
        return np.nan, np.nan, np.nan, np.nan
    xm, ym = max_frame(deconv, xi, yi, rad=rad)
    xc1, yc1 = flux_frame(deconv, xm, ym, rad=subrad)

    if first_coo_is_primary(xc0, yc0, xc1, yc1, dx, dy):
        return xc0, yc0, xc1, yc1
    
    return xc1, yc1, xc0, yc0




def binary_psf(psf_spline, frame, noise, xc0, yc0, xc1, yc1,
                   norm0=1, norm1=1, mask=None, radius=25):
    """Use PSF to simultaneously find the best centers of two stars in
    a frame, using initial guessed positions. init_flux_ratio is star0/star1
    Uses noise frame to clip bad pixels.
    """
    from .reduce import coo_mat
    c_iter = 4
    clip = 10
    xmat0, ymat0 = coo_mat(frame.shape, xc0, yc0)
    xmat1, ymat1 = coo_mat(frame.shape, xc1, yc1)
    xcoo0 =  np.arange(frame.shape[1]) - xc0
    ycoo0 =  np.arange(frame.shape[0]) - yc0
    xcoo1 =  np.arange(frame.shape[1]) - xc1
    ycoo1 =  np.arange(frame.shape[0]) - yc1
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
        psf_frame = (scale0 * psf_spline(xcoo0-dx0, ycoo0-dy0) +
                     scale1 * psf_spline(xcoo1-dx1, ycoo1-dy1))
        psf_pix = psf_frame[sel]
        return np.sum(np.abs((frame_pix-psf_pix)/noise_pix))

    init_param = np.array([0, 0, norm0, 0, 0, norm1])
    for _n in range(c_iter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(chi, init_param)
        psf_frame = (res.x[2] * psf_spline(xcoo0-res.x[0], ycoo0-res.x[1]) +
                     res.x[5] * psf_spline(xcoo1-res.x[3], ycoo1-res.x[4]))
        
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
    Ignore pixels outside of radius from each component.
    """
    from .reduce import coo_mat
    c_iter = 4
    clip = 10
    xmat0, ymat0 = coo_mat(frame.shape, xc0, yc0)
    xmat1, ymat1 = coo_mat(frame.shape, xc0+dx, yc0+dy)
    xcoo0 =  np.arange(frame.shape[1]) - xc0
    ycoo0 =  np.arange(frame.shape[0]) - yc0
    xcoo1 =  np.arange(frame.shape[1]) - xc0 - dx
    ycoo1 =  np.arange(frame.shape[0]) - yc0 - dy
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
        psf_frame = (scale0 * psf_spline(xcoo0-ddx, ycoo0-ddy) +
                     scale1 * psf_spline(xcoo1-ddx, ycoo1-ddy))
        psf_pix = psf_frame[sel]
        return np.sum(np.abs((frame_pix-psf_pix)/noise_pix))

    init_param = np.array([0, 0, norm0, norm1])
    for _n in range(c_iter):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(chi, init_param)
        psf_frame = (res.x[2] * psf_spline(xcoo0-res.x[0], ycoo0-res.x[1]) +
                     res.x[3] * psf_spline(xcoo1-res.x[0]-dx, ycoo1-res.x[1]-dy))
        
        sel = aperture0 * (np.abs(frame - psf_frame) < clip*noise)
        frame_pix = frame[sel]
        noise_pix = noise[sel]
        init_param = res.x
    return res.x
