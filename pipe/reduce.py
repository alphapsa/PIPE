# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:14:41 2020

@author: Alexis Brandeker, alexis@astro.su.se

Collection of tools to manipulate CHEOPS data.

"""

import warnings
import numpy as np
from scipy import interpolate
from scipy.ndimage import shift
from .cent import flux as cent_flux


def resample_imagette_time(sa_time, nexp):
    """Equidistantly distributes nexp exposures
    over the sa_time data points. This function
    is used to correct for a bug in imagette time
    recordings.
    """
    DT = np.median(np.diff(sa_time))
    v = np.linspace(-(nexp-1)/(2*nexp), (nexp-1)/(2*nexp), int(nexp))*DT
    return np.reshape(sa_time[:,None] + v[None,:],-1)


def resample_attitude(obs_mjd, exp_time, attitude):
    """Resamples table of attitude data (RA, DEC, ROLL, DROLL)
    to the midtimes of the observations of duration exp_time
    Note: exptime in seconds, mjd in days,
    DROLL is change in roll angle during exposure
    """
    nrows = obs_mjd.shape[0]
    outparam = np.zeros((nrows, 5))
    outparam[:,0] = obs_mjd
    
    rollin = attitude[:,3].copy()

    for n in range(1,len(rollin)):
        if rollin[n] > rollin[n-1]:
            rollin[n:] -= 360
    
    dt = 0.5*exp_time/24/3600
    t0 = attitude[0,0]
    t = attitude[:,0] - t0
    knots = t[3:-3:5]
    spl = interpolate.LSQUnivariateSpline(t, rollin, knots)
    outparam[:,3] = spl(obs_mjd-t0) % 360
    outparam[:,4] = np.abs(spl(obs_mjd-t0+dt)-spl(obs_mjd-t0-dt))
    
    sample_dt = np.max(np.diff(attitude[:,0]))
    if sample_dt > 2*dt:
        dt = 0.5*sample_dt
    for n in range(nrows):
        t = obs_mjd[n]
        ind = np.abs(attitude[:,0]-t) <= dt
        outparam[n,1] = np.mean(attitude[ind,1]) # RA
        outparam[n,2] = np.mean(attitude[ind,2]) # DEC

    return outparam


def interp_cube(t, t0, datacube0):
    """Resample planes from datacube, no extrapolation
    """
    ifun = interpolate.interp1d(t0, datacube0, axis=0,
                           bounds_error=False, 
                           fill_value=(datacube0[0], datacube0[-1]))
    return ifun(t)


def interp_cube_ext(t, t0, datacube0):
    """Resample planes from datacube, with extrapolation
    """
    tm = np.nanmedian(t)
    ifun = interpolate.interp1d(t0/tm, datacube0, axis=0,
                                copy=True,
                                fill_value='extrapolate',
                                assume_sorted=True)
    return ifun(t/tm)


def coo_mat(shape, xc=0, yc=0):
    """Cordinate matrices defines x,y-coordinate matrices
    wrt to the defined center.
    """
    xcoo = np.arange(shape[0]) - xc
    ycoo = np.arange(shape[1]) - yc
    xmat, ymat = np.meshgrid(xcoo, ycoo)
    return xmat,ymat


def aperture(shape, radius=None, xc=None, yc=None):
    """Returns a boolean 2D-array of the same dimensions as frame
    with True inside radius from (xc, yc), and False outside.
    Used to define disc apertures. Default is center of frame
    and maximum radius inscribing the frame.
    """
    if radius is None:
        radius = int(0.5*np.min(shape))
    if xc is None:
        xc = int(shape[0]/2)
    if yc is None:
        yc = int(shape[1]/2)
    xmat,ymat = coo_mat(shape, xc, yc)
    return xmat**2+ymat**2 < radius**2



def cube_apt(shape, radius, xc, yc):
    """Returns a cube with apertures for each plane along 0-axis
    defined by 1D-arrays xc and yc and constant radius. Used for
    rought photometry since aperture is not weighted along edges.
    """
    xmat,ymat = coo_mat(shape[1:])
    rcube2 = ((np.ones(shape)*xmat[None,:,:]-xc[:,None,None])**2 +
              (np.ones(shape)*ymat[None,:,:]-yc[:,None,None])**2)
    return rcube2 <= radius**2


def replace_nan(data, max_iter = 50):
    """Replaces NaN-entries by mean of neighbours.
    Iterates until all NaN-entries are replaced or
    max_iter is reached. Works on N-dimensional arrays.
    """
    nan_data = data.copy()
    shape = np.append([2*data.ndim], data.shape)
    interp_cube = np.zeros(shape)
    axis = tuple(range(data.ndim))
    shift0 = np.zeros(data.ndim, int)
    shift0[0] = 1
    shift = []
    for n in range(data.ndim):
        shift.append(tuple(np.roll(-shift0, n)))
        shift.append(tuple(np.roll(shift0, n)))
    for _j in range(max_iter):
        for n in range(2*data.ndim):
            interp_cube[n] = np.roll(nan_data, shift[n], axis = axis)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_data = np.nanmean(interp_cube, axis=0)
        nan_data[np.isnan(nan_data)] = mean_data[np.isnan(nan_data)]
        if np.sum(np.isnan(nan_data)) == 0:
            break
    return nan_data


def clean_masked(frame, mask, apt):
    """Cleans all bad pixels as defined by mask inside 
    aperture apt, by averaging neighbouring pixels
    """
    max_iter = 50
    nan_frame = frame.copy()
    nan_frame[mask==0] = np.nan
    interp_cube = np.zeros((4, frame.shape[0], frame.shape[1]))
    shift = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _j in range(max_iter):
        for n in range(4):
            interp_cube[n] = np.roll(nan_frame, shift[n], axis = (0,1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_frame = np.nanmean(interp_cube, axis=0)
        nan_frame[mask==0] = mean_frame[mask==0]
        mask = apt.copy()
        mask[np.isnan(nan_frame)] = 0
        if not np.isnan(np.sum(nan_frame[apt])):
            break
    clean_frame = frame.copy()
    clean_frame[apt] = nan_frame[apt]
    return clean_frame


def clean_cube(data_cube, mask, apt):
    """Cleans all bad pixels as defined by mask (2D or 3D) inside 
    aperture apt, by averaging neighbouring pixels in both
    time and spatial dimensions.
    """
    max_iter = 50
    if mask.ndim == 3:
        mask_cube = mask.copy()
    else:
        mask_cube = np.zeros(data_cube.shape, dtype='?')
        mask_cube[:,:,:] = mask
    nan_cube = data_cube.copy()
    nan_cube[mask_cube==0] = np.nan
    interp_cube = np.zeros((6, data_cube.shape[0],
                            data_cube.shape[1], data_cube.shape[2]))
    shift = [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
             (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for _j in range(max_iter):
        for n in range(6):
            interp_cube[n] = np.roll(nan_cube, shift[n], axis = (0,1,2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_cube = np.nanmean(interp_cube, axis=0)
        nan_cube[mask_cube==0] = mean_cube[mask_cube==0]
        mask_cube[:,:,:] = apt[None, :, :]
        mask_cube[np.isnan(nan_cube)] = 0
        if not np.isnan(np.sum(nan_cube[:, apt])):
            break
    clean_cube = data_cube.copy()
    clean_cube[:, apt] = nan_cube[:, apt]
    return clean_cube


def clean_cube2D(data_cube, mask, apt):
    """Cleans all bad pixels as defined by mask (2D or 3D) inside 
    aperture apt, by averaging neighbouring pixels in
    spatial dimensions (not time).
    """
    max_iter = 50
    if mask.ndim == 3:
        mask_cube = mask.copy()
    else:
        mask_cube = np.zeros(data_cube.shape, dtype='?')
        mask_cube[:,:,:] = mask
    nan_cube = data_cube.copy()
    nan_cube[mask_cube==0] = np.nan
    interp_cube = np.zeros((4, data_cube.shape[0],
                            data_cube.shape[1], data_cube.shape[2]))
    shift = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _j in range(max_iter):
        for n in range(4):
            interp_cube[n] = np.roll(nan_cube, shift[n], axis = (1,2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_cube = np.nanmean(interp_cube, axis=0)
        nan_cube[mask_cube==0] = mean_cube[mask_cube==0]
        mask_cube[:,:,:] = apt[None, :, :]
        mask_cube[np.isnan(nan_cube)] = 0
        if not np.isnan(np.sum(nan_cube[:, apt])):
            break
    clean_cube = data_cube.copy()
    clean_cube[:, apt] = nan_cube[:, apt]
    return clean_cube

    
def check_mask(mask_cube, apt, clip=5, niter=3):
    """Checks the number of masked pixels for each mask of 
    an array, and de-selects those frames that deviate more
    than clip sigma from the average.
    """
    nmask = np.sum(mask_cube[:,apt]==0, axis=1)
    sel = np.ones(nmask.shape, dtype='?')
    for _n in range(niter):
        mmed = np.median(nmask[sel])
        mstd = np.std(nmask[sel])
        sel = nmask-mmed <= clip*mstd
    nbad = np.sum(sel==0)
    if nbad > 0:
        if nbad < 50:
            ret_str = 'Bad frame ({:d}/{:d}):'.format(nbad, len(sel))
            ret_str += '{0}'.format(np.where(sel==0)[0].tolist())
        else: 
            ret_str = 'Bad frames: {:d}/{:d}'.format(nbad, len(sel))
    else:
        ret_str = 'No bad frames out of {:d} detected.'.format(len(sel))
    return sel, ret_str


def check_pos(xc, yc, clip=5, niter=3):
    """Filters positions deviating more than clip
    from the median (iterates niter times). Returns
    the selected frames that were not deviating.
    """
    xc0, yc0 = np.median(xc), np.median(yc)
    r = ((xc-xc0)**2+(yc-yc0)**2)**0.5
    sel = np.ones(r.shape,dtype='?')
    for _n in range(niter):
        s = np.std(r[sel])
        sel = r < clip*s
    nbad = np.sum(sel==0)
    if nbad > 0:
        if nbad < 50:
            ret_str = 'Bad offsets, lim {:.2f} ({:d}/{:d}):\n'.format(
                        clip*s, nbad, len(sel))
            ret_str += '{0}'.format(np.where(sel==0)[0].tolist())
        else: 
            ret_str = 'Bad offsets, lim {:.2f}: {:d}/{:d}'.format(
                        clip*s, nbad, len(sel))
    else:
        ret_str = 'No bad offsets out of {:d}, lim {:.2f}, detected.'.format(
                    len(sel), clip*s)
    return sel, ret_str


def check_motion(xc, lowfrac = 0.5):
    """Selects frames with positions after and before that are fairly 
    consistent, with the goal to select frames with little 
    motion blur. lowfrac is the fraction of frames to keep.
    """
    sel = np.ones(xc.shape, dtype='?')
    X = np.array([xc[:-2],xc[1:-1], xc[2:]])
    s = np.std(X, axis=0)
    level = np.percentile(s, 100*lowfrac)
    sel[1:-1] = (s < level)
    sel[0], sel[-1] = sel[1], sel[-2]
    ret_str = 'Motion filter ({:.0f}%, {:d} frames): PEAK {:.2f} pix, AVG {:.2f} pix'.format(
            lowfrac*100, np.sum(sel), level, np.mean(s[sel[1:-1]]))
    return sel, ret_str


def check_val(val, clip=5, niter=3):
    """Filters values in array deviating more than clip
    from the average (iterates niter times). Returns
    the selected frames that were not deviating.
    """
    sel = np.ones(val.shape,dtype='?')
    val0 = np.median(val)
    for _n in range(niter):
        s = np.std(val[sel])
        val0 = np.median(val[sel])        
        sel = np.abs(val-val0) < clip*s
    nbad = np.sum(sel==0)
    if nbad > 0:
        if nbad < 50:
            ret_str = 'Bad values ({:d}/{:d}):'.format(nbad, len(sel))
            ret_str += '{0}'.format(np.where(sel==0)[0].tolist())
        else: 
            ret_str = 'Bad values: {:d}/{:d}'.format(nbad, len(sel))
    else:
        ret_str = 'No bad values out of {:d} detected.'.format(len(sel))
    return sel, ret_str


def check_low(val, clip=5, niter=3):
    """Filters values lower more than clip
    from the average (iterates niter times). Returns
    the selected frames that were not deviating.
    """
    sel = np.ones(val.shape,dtype='?')
    val0 = np.median(val)
    for _n in range(niter):
        s = np.std(val[sel])
        val0 = np.median(val[sel])        
        sel = ((val0-val) < clip*s)
    nbad = np.sum(sel==0)
    if nbad > 0:
        if nbad < 50:
            ret_str = 'Source missing ({:d}/{:d}):'.format(nbad, len(sel))
            ret_str += '{0}'.format(np.where(sel==0)[0].tolist())
        else: 
            ret_str = 'Source missing: {:d}/{:d}'.format(nbad, len(sel))
    else:
        ret_str = 'No frames {:d} with source missing.'.format(len(sel))
    return sel, ret_str


def rough_contrast(cube, radius=25, mask=None):
    """Computes a rouch contrast (flux difference) between pixels
    inside aperture and outside aperture, for each frame. Used
    to determine of source is missing from image or not.
    """
    if mask is None:
        xc, yc = cent_flux(np.abs(cube))
    else:
        xc, yc = cent_flux(np.abs(cube)*mask)
    apt_in = cube_apt(cube.shape, radius, xc, yc)
    apt_out = (apt_in==0)
    if mask is not None:
        apt_in *= mask[None,:,:]
        apt_out *= mask[None,:,:]
    inmed = np.zeros(len(cube))
    outmed = np.zeros(len(cube))
    for n in range(len(cube)):
        inmed[n] = np.percentile(cube[n,apt_in[n]], 90)
        outmed[n] = np.percentile(cube[n,apt_out[n]], 90)
    return inmed-outmed
    

def noise(debias, ron_elec, e_per_ADU=1):
    """ Computes the photon + read-out noise of a bias 
    subtracted frame (or cube).
    """
    return (np.abs(debias)*e_per_ADU + ron_elec**2)**0.5 / e_per_ADU


def psf_noise(source_model, ron_elec, e_per_ADU=1):
    """ Uses a source model (PSF*scale+background) of the source 
    to compute the photon + read-out noise.
    """
    return (np.abs(source_model)*e_per_ADU + ron_elec**2)**0.5 / e_per_ADU


def empiric_noise(residual_cube, xc, yc, bg=None, niter=10, sigma_clip=3):
    """Checks the consistency of each plane in cube, offset
    to proper centre the PSFs. The resulting statistical
    noise is then offset back to original position, and a
    cube is returned with the empirical noise for each pixel
    and plane. The underlying assumption is that the noise
    is constant throughout the time line; e.g. varying
    background noise has to be added.
    """
    shift_cube = np.zeros_like(residual_cube)
    noise_cube = np.zeros_like(residual_cube)
    xm = np.nanmedian(xc)
    ym = np.nanmedian(yc)
    dx = xc - xm
    dy = yc - ym
    for n in range(len(residual_cube)):
        shift_cube[n] = shift(residual_cube[n], (-dy[n], -dx[n]), order=1)
    shift_cube -= np.nanmedian(shift_cube, axis=0)
    Nsigma = sigma_clip*np.nanmax(np.abs(shift_cube), axis=0)
    for n in range(niter):
        shift_cube0 = shift_cube.copy()
        shift_cube0[np.greater(np.abs(shift_cube), Nsigma[None,:,:])] = np.nan
        Nsigma = sigma_clip * np.nanstd(shift_cube0, axis=0)
    for n in range(len(residual_cube)):
        noise_cube[n] = shift(Nsigma, (dy[n], dx[n]), order=1)
    if bg is not None:
        bgm = np.nanmedian(bg)
        bg_noise2 = bg - bgm
        noise_cube = np.abs(noise_cube**2 + bg_noise2[:,None, None])**.5
    return noise_cube


def integrate_psf(psf_fun, radius=23):
    """Computes the flux of the psf inside radius. Used
    to normalise the PSF.
    """
    x = np.linspace(-radius, radius, 2*radius+1)
    psf_mat = psf_fun(x,x)
    apt = aperture(psf_mat.shape, radius=radius)
    return np.sum(apt*psf_mat)

    
def pix_mat(datacube, noisecube, xc, yc, mask, fscale=None):
    """Turn a data cube into a pixel matrix with columns
    y-coordinate, x-coordinate, pixel value, pixel value noise
    the mask can either be a 2D plane or 3D cube (True = select,
    False mask out).
    """
    xmat, ymat = coo_mat(datacube[0].shape)
    N = len(datacube)   # Number of frames
    npix = np.sum(mask) # Number of pixels per frame
    if mask.ndim == 3:
        totpix = np.sum(mask)
    else:
        totpix = N*np.sum(mask)
        mask0 = mask
    n0 = 0
    npix = np.sum(mask)
    pix = np.zeros((totpix, 5))

    if fscale is None:
        fscale = np.ones(len(datacube))

    for n in range(N):
        if mask.ndim == 3:
            mask0 = mask[n]
            npix = np.sum(mask0)
        n1 = n0 + npix
        pix[n0:n1, 0] = ymat[mask0]-yc[n]
        pix[n0:n1, 1] = xmat[mask0]-xc[n]
        pix[n0:n1, 2] = datacube[n][mask0]/fscale[n]
        pix[n0:n1, 3] = noisecube[n][mask0]/fscale[n]
        n0 = n1
    return pix


def make_maskcube(data_cube, noise_cube, model_cube,
                  mask=None, clip=5):
    """Find pixels that deviate too much from fitted models, and
    produce a cube of masks.
    """
    mask_cube = np.ones(data_cube.shape, dtype='?')
    mask_cube[np.greater(np.abs(data_cube - model_cube), clip*noise_cube)] = 0
    if mask is not None:
        mask_cube *= mask
    return mask_cube        
