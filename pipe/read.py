# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:00:21 2020

@author: Alexis Brandeker, alexis@astro.su.se

A collection of routines that read data from CHEOPS fits format
files, and typically returns numpy arrays.

Also contains routines that save data in fits or text formats.

"""

import os
import numpy as np
from scipy import interpolate
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord, get_body_barycentric
from .spline_pca import make_spline2D
from .pipe_statistics import sigma_clip

# Copy data from fits-files into memory
fits.Conf.use_memmap = False

def raw_datacube(filename, frame_range=None):
    """Read CHEOPS raw datacube format, either subarray or imagettes.
    Returns cube as numpy array where the first index is frame
    number, an array with the mjd for each frame, the header
    and an associated table with various pre-frame data
    contained in the fits-files, like e.g. the bias values.
    Introduces np.nan values for array elements without data
    (e.g. outside circular boundary).
    """
    with fits.open(filename) as hdul:
        rawcube = np.array(hdul[1].data, dtype='f8')
#        np.nan_to_num(rawcube, copy=False)
        rawcube[rawcube==0] = np.nan
        hdr = hdul[0].header + hdul[1].header
        if len(hdul) < 9: # Imagettes
            mjd = hdul[2].data['MJD_TIME']
            tab = hdul[2].data
        else:  # Raw subarray file
            mjd = hdul[9].data['MJD_TIME']
            tab = hdul[9].data
    if frame_range is not None:
        return (rawcube[frame_range[0]:frame_range[1]], 
               mjd[frame_range[0]:frame_range[1]], 
               hdr, 
               tab[frame_range[0]:frame_range[1]])
    return rawcube, mjd, hdr, tab


def lightcurve(filename):
    """Reads the DRP (or PIPE) lightcurve fits file,
         returns a numpy dict table
    """
    with fits.open(filename) as hdul:
        lc = hdul[1].data
    return lc


def fits_cube(filename, level=0):
    """Reads raw fits cube, returns data (converted to doubles) and header.
    level is the fits-level of the data, in case of multiple fits layers.
    """
    with fits.open(filename) as hdul:
        cube = np.array(hdul[level].data, dtype='f8')
        hdr = hdul[level].header.copy()
    return cube, hdr


def nonlinear(filename):
    """Reads the non-linear correction from a numpy array file with
    ADU vs multiplicative correction. The correction should be
    applied after bias subtraction. Return is an interpolation
    function that gives correction as a function of ADU.
    """
    nl = np.load(filename)
    ifun = interpolate.interp1d(nl[:, 0], nl[:, 1], axis=0,
                                bounds_error=False,
                                fill_value=(nl[0, 1], nl[-1, 1]))
    return ifun


def attitude(filename):
    """Reads the CHEOPS attitude file and puts the data into
    a N-by-4 array with spacecraft mjd, ra, dec, and roll angle.
    """
    with fits.open(filename) as hdul:
        outparam = np.zeros((hdul[1].header['NAXIS2'], 4))
        outparam[:, 0] = hdul[1].data['MJD_TIME']
        outparam[:, 1] = hdul[1].data['SC_RA']
        outparam[:, 2] = hdul[1].data['SC_DEC']
        outparam[:, 3] = hdul[1].data['SC_ROLL_ANGLE']
    return outparam


def starcat(filename, colstr, entry=0):
    """Reads star catalogue file and returns value for
    column string colstr and entry row
    """
    with fits.open(filename) as hdul:
        val = hdul[1].data[colstr][entry]
    return val


def raw_param(filename, data_index, param_name):
    """Reads the specific sensor from the CHEOPS sa raw file.
    """
    with fits.open(filename) as hdul:
        ret_param = np.asarray(hdul[data_index].data[param_name])
    return ret_param


def bias_ron_adu(filename, gain):
    """Estimates bias and read-out noise (in electrons) from raw subarray file.
    gain is in electrons per ADU
    """
    with fits.open(filename) as hdul:
        nexp = hdul[1].header['NEXP']
        bias_pix = gain*np.array(hdul[2].data.flat, dtype='f8')
        sel = sigma_clip(bias_pix, clip=3, niter=10)
        bias = np.nanmedian(bias_pix[sel])/nexp
        ron = np.nanstd(bias_pix[sel])/nexp**.5
    return bias, ron


def PSFs(psf_files, psf_ref_path):
    """From a list of filenames of PSF spline parameter files,
    load them and put the content into a list to be returned.
    """
    psflist = []
    for filename in psf_files:
        filepath = os.path.join(psf_ref_path, filename)
        spl_params = np.load(filepath, allow_pickle=True)
        psflist.append(make_spline2D(spl_params))
    return psflist        


def gain(file_hk, file_gain):
    """Compute gain using HK parameters and the gain reference table
    Returns gain in units of e/ADU sampled at times MJD
    """
    with fits.open(file_hk) as hdul:
        data = hdul[1].data
        volt_vod = data['VOLT_FEE_VOD']
        volt_vrd = data['VOLT_FEE_VRD']
        volt_vog = data['VOLT_FEE_VOG']
        volt_vss = data['VOLT_FEE_VSS']
        temp_ccd = data['VOLT_FEE_CCD']
        mjd = data['MJD_TIME']

    with fits.open(file_gain) as hdul:
        data = hdul[1].data
        hdr = hdul[1].header
        vod_off = hdr['VOD_OFF']
        vrd_off = hdr['VRD_OFF']
        vog_off = hdr['VOG_OFF']
        vss_off = hdr['VSS_OFF']
        temp_off = hdr['TEMP_OFF']
        gain_nom = hdr['GAIN_NOM']
        gain_fact = data['FACTOR']
        exp_vod = data['EXP_VOD']
        exp_vrd = data['EXP_VRD']
        exp_vog = data['EXP_VOG']
        exp_vss = data['EXP_VSS']
        exp_temp = data['EXP_TEMP']

    gain_vec = gain_nom * (1 + np.sum(gain_fact[None, :] *
                                      (volt_vss[:, None] - vss_off) ** exp_vss[
                                                                       None,
                                                                       :] *
                                      (volt_vod[:, None] - volt_vss[:,
                                                           None] - vod_off) ** exp_vod[
                                                                               None,
                                                                               :] *
                                      (volt_vrd[:, None] - volt_vss[:,
                                                           None] - vrd_off) ** exp_vrd[
                                                                               None,
                                                                               :] *
                                      (volt_vog[:, None] - volt_vss[:,
                                                           None] - vog_off) ** exp_vog[
                                                                               None,
                                                                               :] *
                                      (temp_ccd[:,
                                       None] + temp_off) ** exp_temp[None, :],
                                      axis=1))

    return mjd, 1 / gain_vec


def thermFront_2(filename):
    """Reads frontTemp_2 sensor data from the CHEOPS raw file.
    """
    return raw_param(filename, data_index=9, param_name='thermFront_2')



def mjd2bjd(mjd, ra, dec):
    """Compute BJD given MJD and direction. The observer is assumed
    to be located at Earth centre, giving a maximum error of 23 ms.
    mjd can be an array of MJD dates. ra and dec in degrees.
    """
    t = Time(mjd, format='mjd')
    r = get_body_barycentric('earth', t)
    n = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs').cartesian
    
    bjd = mjd + 2400000.5 + (n.dot(r)/const.c).to_value(u.d)
    return bjd


def sub_image_indices(offset, shape):
    """Helper function that computes index ranges
    given a 2D offset and a 2D shape
    """
    i0 = int(offset[0])
    i1 = int(i0 + shape[0])
    j0 = int(offset[1])
    j1 = int(j0 + shape[1])
    return i0, i1, j0, j1


def flatfield(filename, Teff, offset, shape):
    """Reads the flatfield cube and interpolates the
    flatfield temperatures to the given temperature.
    The part of the detector that is returned is defined
    by the offset and size (in 2D pixel coordinates).
    """
    with fits.open(filename) as hdul:
        T = hdul[2].data['T_EFF']
        T = T[hdul[2].data['DATA_TYPE'] == 'FLAT FIELD']
        idx = np.searchsorted(T, Teff, side="left")
        a = (Teff - T[idx]) / (T[idx + 1] - T[idx])
        i0, i1, j0, j1 = sub_image_indices(offset, shape)
        ff0 = hdul[1].data[idx, j0:j1, i0:i1]
        ff1 = hdul[1].data[idx + 1, j0:j1, i0:i1]
    return ff0 * (1 - a) + ff1 * a


def find_brack_ind(data, v):
    """Finds indices n0 and n1 such that 
    vec[n0] and vec[n1] are the closest bracketing
    values in array data to value v. Values in data
    need not be ordered. If value is outside range of 
    values in array, then the index of the nearest 
    array value is returned (twice).
    """
    if np.min(data) < v:
        diff = v-data
        diff[diff<0] = np.nan
        n0 = np.nanargmin(diff)
    else: 
        n0 = np.argmin(data)
        return n0, n0
    if np.max(data) > v:
        diff = data-v
        diff[diff<0] = np.nan
        n1 = np.nanargmin(diff)
    else: 
        return n0, n0
    return n0, n1


def dark(darkpath, mjd, offset, shape):
    """Traverses darkpath directory, looking for all
    dark current files and interpolates the two 
    closest in time.
    """
    darkfiles = []; mjds = []
    for root, _dirs, files in os.walk(darkpath):
        for file in files:
            if 'REF_APP_DarkFrame' in file:
                filename = os.path.join(root, file)
                darkfiles.append(filename)
                with fits.open(filename) as hdul:
                    mjds.append(Time(hdul[1].header['V_STRT_U'],
                                scale='tt', format='isot').tt.mjd)
    if len(mjds) < 1:
        raise ValueError('Missing dark frame reference file. You may turn off '
                         'this feature by setting `pps.darksub = False`.')
    
    n0, n1 = find_brack_ind(np.array(mjds), mjd)
    i0, i1, j0, j1 = sub_image_indices(offset, shape)

    with fits.open(darkfiles[n0]) as hdul:
        dark0 = hdul[1].data[0, j0:j1, i0:i1].copy()
        dark_err0 = hdul[1].data[1, j0:j1, i0:i1].copy()
    if n0 == n1:
        return dark0, dark_err0, darkfiles[n0], darkfiles[n1]
    with fits.open(darkfiles[n1]) as hdul:
        dark1 = hdul[1].data[0, j0:j1, i0:i1].copy()
        dark_err1 = hdul[1].data[1, j0:j1, i0:i1].copy()

    t = (mjd-mjds[n0])/(mjds[n1]-mjds[n0])
    dark = (1-t)*dark0 + t*dark1
    dark_err = ((1-t)**2*dark_err0**2 + t**2*dark_err1**2)**0.5
    return dark, dark_err, darkfiles[n0], darkfiles[n1]


def bad(badpath, mjd, offset, shape):
    """Traverses badpath directory, looking for all
    bad pixel map current files and selects the nearest
    in time.
    """
    badfiles = []; mjds = []
    for root, _dirs, files in os.walk(badpath):
        for file in files:
            if 'REF_APP_BadPixelMap' in file:
                filename = os.path.join(root, file)
                badfiles.append(filename)
                with fits.open(filename) as hdul:
                    mjds.append(Time(hdul[1].header['V_STRT_U'],
                                scale='tt', format='isot').tt.mjd)
    if len(mjds) < 1:
        raise ValueError('Missing bad pixel map reference file. You may turn off '
                         'this feature by setting `pps.mask_badpix = False`.')
    
    n = np.argmin(np.abs(mjd-np.array(mjds)))

    i0, i1, j0, j1 = sub_image_indices(offset, shape)
    with fits.open(badfiles[n]) as hdul:
        bad = hdul[1].data[j0:j1, i0:i1].copy()
    return bad, badfiles[n]


def imagette_offset(filename, frame_range=None):
    """Returns the first imagette offset from an
    imagette fits-file cube; first offset is relative
    to full array, second offset is relative to subarray
    """
    with fits.open(filename) as hdul:
        x_off = hdul[2].data['X_OFF_FULL_ARRAY'][0]
        y_off = hdul[2].data['Y_OFF_FULL_ARRAY'][0]
        x_sa_off = hdul[2].data['X_OFF_SUB_ARRAY'][0]
        y_sa_off = hdul[2].data['Y_OFF_SUB_ARRAY'][0]
    return (x_off, y_off), (x_sa_off, y_sa_off)
    # raise Exception('[imagette_offset] Error: {:s} not found'.format(filename))


def save_cube_fits(filename, cube):
    """Save a data cube as a fits file.
    """
    hdu = fits.PrimaryHDU(cube)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)


def save_bg_star_phot_fits(filename, t, bjd, fluxes, gaia_IDs, header):
    """Save lightcurve data of fitted background stars, as defined by
    arguments, to fits table in binary format. The stars are identified by 
    their Gaia ID numbers in the comment for respective fits-column
    """
    c = []
    c.append(fits.Column(name='MJD_TIME', format='D', unit='day', array=t))
    c.append(fits.Column(name='BJD_TIME', format='D', unit='day', array=bjd))
    for n in range(fluxes.shape[1]):
        c.append(fits.Column(name='f{:d}'.format(n), format='D', unit='electrons',
                             array=fluxes[:, n]))
    tab = fits.BinTableHDU.from_columns(c, header=header)
    for n in range(fluxes.shape[1]):
        key = f'TTYPE{n+3}'
        tab.header[key] = (tab.header[key], f'{gaia_IDs[n]}')
    tab.writeto(filename, overwrite=True, checksum=True)


def save_eigen_fits(filename, t, bjd, sc, err, bg, roll, xc, yc, flag,
                    w, thermFront_2, header):
    """Save lightcurve data as defined by arguments to fits table in binary
    format. Coefficients for the principle components of the PSF eigen
    analysis are also added, as well as the thermFront_2 values, to be
    used in de-correlations.
    """
    c = []
    c.append(fits.Column(name='MJD_TIME', format='D', unit='day', array=t))
    c.append(fits.Column(name='BJD_TIME', format='D', unit='day', array=bjd))
    c.append(fits.Column(name='FLUX', format='D', unit='electrons', array=sc))
    c.append(
        fits.Column(name='FLUXERR', format='D', unit='electrons', array=err))
    c.append(fits.Column(name='BG', format='D', unit='electrons/pix', array=bg))
    c.append(fits.Column(name='ROLL', format='D', unit='deg', array=roll))
    c.append(fits.Column(name='XC', format='D', unit='pix', array=xc))
    c.append(fits.Column(name='YC', format='D', unit='pix', array=yc))
    c.append(fits.Column(name='FLAG', format='I', array=flag))
    for n in range(w.shape[1]):
        c.append(fits.Column(name='U{:d}'.format(n), format='D', array=w[:, n]))
    c.append(fits.Column(name='thermFront_2', format='D', array=thermFront_2))
    tab = fits.BinTableHDU.from_columns(c, header=header)
    tab.writeto(filename, overwrite=True, checksum=True)


def save_binary_eigen_fits(filename, t, bjd, sc0, sc1, bg, roll,
                           xc0, yc0, xc1, yc1, flag,
                           w0, w1, thermFront_2, header):
    """Save lightcurve data from both componentes of a binary, as defined
    by arguments to fits table in binary format. Coefficients for the
    principle components of both stars from the PSF eigen analysis are
    also added, as well as the thermFront_2 values, to be used in
    de-correlations.
    """
    c = []
    c.append(fits.Column(name='MJD_TIME', format='D', unit='day', array=t))
    c.append(fits.Column(name='BJD_TIME', format='D', unit='day', array=bjd))
    c.append(fits.Column(name='FLUX0', format='D', unit='electrons', array=sc0))
    c.append(fits.Column(name='FLUX1', format='D', unit='electrons', array=sc1))
    c.append(fits.Column(name='BG', format='D', unit='electrons/pix', array=bg))
    c.append(fits.Column(name='ROLL', format='D', unit='deg', array=roll))
    c.append(fits.Column(name='XC0', format='D', unit='pix', array=xc0))
    c.append(fits.Column(name='YC0', format='D', unit='pix', array=yc0))
    c.append(fits.Column(name='XC1', format='D', unit='pix', array=xc1))
    c.append(fits.Column(name='YC1', format='D', unit='pix', array=yc1))
    c.append(fits.Column(name='FLAG', format='I', array=flag))
    for n in range(w0.shape[1]):
        c.append(
            fits.Column(name='U{:d}'.format(n), format='D', array=w0[:, n]))
    for n in range(w1.shape[1]):
        c.append(
            fits.Column(name='W{:d}'.format(n), format='D', array=w1[:, n]))
    c.append(fits.Column(name='thermFront_2', format='D', array=thermFront_2))
    tab = fits.BinTableHDU.from_columns(c, header=header)
    tab.writeto(filename, overwrite=True, checksum=True)


def save_txt(filename, t, flux, err, bg, roll, xc, yc):
    """Save lightcurve to textfile according to arrays
    defined by arguments
    """
    X = np.array([t, flux, err, bg, roll, xc, yc]).T
    fmt = '%26.18e'
    np.savetxt(filename, X=X, fmt=fmt)


def save_psf_filenames(filename, psf_filenames):
    np.savetxt(filename, psf_filenames, fmt='%s')

def read_psf_filenames(filename):
    with open(filename, 'r') as fp:
        return [filename.strip() for filename in fp.readlines()]
