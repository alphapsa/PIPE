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


def datacube(filename, frame_range = None):
    """Read CHEOPS datacube format, either subarray or imagettes.
    Returns cube as numpy array where the first index is frame
    number, an array with the mjd for each frame, the header
    and an associated table with various pre-frame data 
    like e.g. the bias values.
    """
    with fits.open(filename) as hdul:
        rawcube = np.array(hdul[1].data, dtype='f8')
        np.nan_to_num(rawcube, copy=False)
        hdr = hdul[0].header + hdul[1].header
        mjd = hdul[2].data['MJD_TIME']
        tab = hdul[2].data
    if frame_range is not None:
        return rawcube[frame_range[0]:frame_range[1]], \
                mjd[frame_range[0]:frame_range[1]], \
                hdr, \
                tab[frame_range[0]:frame_range[1]]
    return rawcube, mjd, hdr, tab


def lightcurve(filename):
    """Reads the DRP (or PIPE) lightcurve fits file,
         returns a numpy dict table
    """
    with fits.open(filename) as hdul:
        return hdul[1].data
    raise Exception('[lightcurve] Error: \"{:s}\" not found'.format(filename))


def fits_cube(filename, level=0):
    """Reads raw fits cube, returns data (converted to doubles) and header.
    level is the fits-level of the data, in case of multiple fits layers.
    """
    with fits.open(filename) as hdul:
        cube = np.array(hdul[level].data, dtype='f8')
        hdr = hdul[level].header
    return cube, hdr


def mask(filename):
    """Reads a predefined mask saved as a fits file,
        returns numpy array
    """
    with fits.open(filename) as hdul:
        return hdul[1].data
    raise Exception('[mask] Error: {:s} not found'.format(filename))


def nonlinear(filename):
    """Reads the non-linear correction from a text file with 
    ADU vs multiplicative correction. The correction should be 
    applied after bias subtraction. Return is an interpolation
    function that gives correction as a function of ADU.
    """
    nl = np.loadtxt(filename)
    ifun = interpolate.interp1d(nl[:,0], nl[:,1], axis=0,
                       bounds_error = False, 
                       fill_value = (nl[0,1],nl[-1,1]))
    return ifun
    

def attitude(filename):
    """Reads the CHEOPS attitude file and puts the data into
    a N-by-4 array with spacecraft mjd, ra, dev, and roll angle.
    """
    with fits.open(filename) as hdul:
        outparam = np.zeros((hdul[1].header['NAXIS2'], 4))
        outparam[:,0] = hdul[1].data['MJD_TIME']
        outparam[:,1] = hdul[1].data['SC_RA']
        outparam[:,2] = hdul[1].data['SC_DEC']
        outparam[:,3] = hdul[1].data['SC_ROLL_ANGLE']
        return outparam
    raise Exception('[attitude] Error: {:s} not found'.format(filename))


def starcat(filename, colstr, entry=0):
    """Reads star catalogue file and returns value for
    column string colstr and entry row
    """
    with fits.open(filename) as hdul:
        return hdul[1].data[colstr][entry]
    raise Exception('[starcat] Error: {:s} not found'.format(filename))
    

def raw_param(filename, data_index, param_name):
    """Reads the specific sensor from the CHEOPS sa raw file.
    """
    with fits.open(filename) as hdul:
        ret_param = hdul[data_index].data[param_name].copy()
        return ret_param
    raise Exception('[raw_param] Error: {:s} not found'.format(filename))


def ron(filename):
    """Reads the readout noise table entry from the CHEOPS
    calibratared subarray cube file. Returns a single value (in ADU)
    """
    return np.nanmedian(raw_param(filename, data_index=2, param_name='RON'))


def gain(file_hk, file_gain):
    """Compute gain using HK parameters and the gain reference table
    Returns gain in units of e/ADU
    """
    with fits.open(file_hk) as hdul:
        data = hdul[1].data
        volt_vod = data['VOLT_FEE_VOD']
        volt_vrd = data['VOLT_FEE_VRD']
        volt_vog = data['VOLT_FEE_VOG']
        volt_vss = data['VOLT_FEE_VSS']
        temp_ccd = data['VOLT_FEE_CCD']

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

    gain_vec = gain_nom*(1 + np.sum(gain_fact[None, :] * 
               (volt_vss[:, None]-vss_off)**exp_vss[None, :] *
               (volt_vod[:, None]-volt_vss[:, None]-vod_off)**exp_vod[None, :] *
               (volt_vrd[:, None]-volt_vss[:, None]-vrd_off)**exp_vrd[None, :] *
               (volt_vog[:, None]-volt_vss[:, None]-vog_off)**exp_vog[None, :] *
               (temp_ccd[:, None]+temp_off)**exp_temp[None, :], axis=1))
    return 1/gain_vec
    


def thermFront_2(filename):
    """Reads frontTemp_2 sensor data from the CHEOPS raw file.
    """
    return raw_param(filename, data_index=9, param_name='thermFront_2')


def mjd2bjd(filename):
    with fits.open(filename) as hdul:
        mjd = hdul[2].data['MJD_TIME']
        bjd = hdul[2].data['BJD_TIME']
        ifun = interpolate.interp1d(mjd, bjd,
                           bounds_error = False, 
                           fill_value = 'extrapolate')
        return ifun
    raise Exception('[mjd2bjd] Error: {:s} not found'.format(filename))


def sub_image_indices(offset, size):
    """Helper function that computes index ranges
    given a 2D offset and a 2D size
    """
    i0 = offset[0]
    i1 = i0+size[0]
    j0 = offset[1]
    j1 = j0+size[1]
    return i0, i1, j0, j1


def flatfield(filename, Teff, offset, size):
    """Reads the flatfield cube and interpolates the
    flatfield temperatures to the given temperature.
    The part of the detector that is returned is defined
    by the offset and size (in 2D pixel coordinates).
    """
    with fits.open(filename) as hdul:
        T = hdul[2].data['T_EFF']
        T = T[hdul[2].data['DATA_TYPE']=='FLAT FIELD']
        idx = np.searchsorted(T, Teff, side="left")
        a = (Teff-T[idx])/(T[idx+1]-T[idx])
        i0, i1, j0, j1 = sub_image_indices(offset, size)
        ff0 = hdul[1].data[idx, j0:j1, i0:i1]
        ff1 = hdul[1].data[idx+1, j0:j1, i0:i1]
        return ff0*(1-a) + ff1*a
    raise Exception('[flatfield] Error: {:s} not found'.format(filename))


def dark(darkpath, mjd, offset, size):
    """Traverses darkpath directory, looking for all 
    dark current files and picks the one closest in time
    """
    darkfiles = []
    mjds = []
    for root, dirs, files in os.walk(darkpath):
        for file in files:
            if 'MCO_REP_DarkFrameFullArray' in file:
                filename = os.path.join(root, file)
                darkfiles.append(filename)
                with fits.open(filename) as hdul:
                    mjds.append(hdul[1].header['V_STRT_M'])
    ind = np.argmin(np.abs(np.array(mjds)-mjd))
    i0, i1, j0, j1 = sub_image_indices(offset, size)

    with fits.open(darkfiles[ind]) as hdul:
        dark = hdul[1].data[0, j0:j1, i0:i1]
        dark_err = hdul[1].data[1, j0:j1, i0:i1]
        return dark, dark_err
    return np.zeros(size), np.zeros(size)


def imagette_offset(filename, frame_range = None):
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
    raise Exception('[imagette_offset] Error: {:s} not found'.format(filename))



def save_eigen_fits(filename, t, bjd, sc, err, bg, roll, xc, yc, flag, chi2,
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
    c.append(fits.Column(name='FLUXERR', format='D', unit='electrons', array=err))
    c.append(fits.Column(name='BG', format='D', unit='electrons/pix', array=bg))
    c.append(fits.Column(name='CHI2', format='D', array=chi2))
    c.append(fits.Column(name='ROLL', format='D', unit='deg', array=roll))
    c.append(fits.Column(name='XC', format='D', unit='pix', array=xc))
    c.append(fits.Column(name='YC', format='D', unit='pix', array=yc))
    c.append(fits.Column(name='FLAG', format='I', array=flag))
    for n in range(w.shape[1]):
        c.append(fits.Column(name='U{:d}'.format(n), format='D', array=w[:,n]))
    c.append(fits.Column(name='thermFront_2', format='D', array=thermFront_2))
    tab = fits.BinTableHDU.from_columns(c, header=header)
    tab.writeto(filename, overwrite=True, checksum=True)


def save_binary_eigen_fits(filename, t, bjd, sc0, sc1, bg, roll, 
                           xc0, yc0, xc1, yc1, flag, chi2,
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
    c.append(fits.Column(name='CHI2', format='D', array=chi2))
    c.append(fits.Column(name='ROLL', format='D', unit='deg', array=roll))
    c.append(fits.Column(name='XC0', format='D', unit='pix', array=xc0))
    c.append(fits.Column(name='YC0', format='D', unit='pix', array=yc0))
    c.append(fits.Column(name='XC1', format='D', unit='pix', array=xc1))
    c.append(fits.Column(name='YC1', format='D', unit='pix', array=yc1))
    c.append(fits.Column(name='FLAG', format='I', array=flag))
    for n in range(w0.shape[1]):
        c.append(fits.Column(name='U{:d}'.format(n), format='D', array=w0[:,n]))
    for n in range(w1.shape[1]):
        c.append(fits.Column(name='W{:d}'.format(n), format='D', array=w1[:,n]))
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

