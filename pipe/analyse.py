# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:00:44 2020

@author: Alexis Brandeker, alexis@astro.su.se

A collection of routines and tools that analyse both extracted
and non-extracted data in various ways, including rudimentary aperture
photometry.

"""
import numpy as np
import os
from scipy.interpolate import LSQUnivariateSpline
from .read import lightcurve


def mad(series):
    """Compute normalised mean absolute deviation (MAD) for an
    array, return in ppm
    """
    return np.nanmedian(np.abs(np.diff(series/np.nanmedian(series))))*1e6


def find_orbits(roll, phase=None, sub_orbits=1):
    """Finds the indices for each orbit repeat, 
    starting with the first roll angle. Assumes that 
    roll angles decrease, and that part of the orbit 
    has roll angles higher than the first one. Orbits can
    start at a different phase than the first element,
    if defined.
    sub_orbits defines how many suborbits each orbit
    should be divided to.
    """
    orbit_index = []
    if phase is None:
        orbit_index.append(0)
        phase0 = roll[0]
    else:
        phase0 = phase
    switch = False
    for n in range(1,len(roll)):
        if switch and roll[n] <= phase0:
            orbit_index.append(n)
            switch = False             
        switch =  roll[n] > roll[n-1] or switch
    if len(orbit_index) <= 2:
        orbit_index = (0, len(roll)-1)
        
    if sub_orbits > 1:
        suborb_list = []
        dorb = np.array(np.diff(orbit_index)/sub_orbits, dtype='int')
        for n in range(len(orbit_index)-1):
            for m in range(sub_orbits):
                suborb_list.append(orbit_index[n]+m*dorb[n])
        suborb_list.append(orbit_index[-1])
        orbit_index = np.array(suborb_list)
        
    return orbit_index
        
    
def psf_phot_cube(data_cube, noise_cube, psf_cube, bg, apt):
    """Use defined PSF to integrate flux and derive error. Only
    region inside apt is used (can be cube with apt defined for
    each frame, or single image for the same region in every frame
    to be used).
    """
    N = len(data_cube)
    f = np.zeros(N)
    e = np.zeros(N)
    if apt.ndim == 2:
        sapt = apt
    for n in range(N):
        if apt.ndim > 2:
            sapt = apt[n]
        frame = data_cube[n]-bg[n]
        f[n], e[n] = phot(frame, noise_cube[n], psf_cube[n], sapt)
    return f, e


def phot(frame, noise, psf, apt):
    """Use defined PSF to integrate flux and derive error. Only
    region inside apt is used.
    """
    npsf = np.abs(psf[apt])
    s1 = np.sum(npsf*frame[apt]/noise[apt]**2)
    s2 = np.sum(npsf**2/noise[apt]**2)
    if s2 > 0:
        return s1/s2, (1/s2)**0.5
    print('WARNING [analyse/phot]: Empty flux integration encountered')
    return 0, -1


def apt_phot_cube(data_cube, noise_cube, radius, xc, yc):
    """Do aperture photometry on data cube, assuming the background
    has already been subtracted. A circular aperture of radius centered
    on xc, yc is used with a linearly interpolated edge. Returns the
    flux and its std noise as arrays.
    """
    flux = np.zeros(len(data_cube))
    noise = flux.copy()
    apt_coo = np.linspace(-radius,radius,2*radius+1)
    xmat, ymat = np.meshgrid(apt_coo, apt_coo)
    apt = xmat**2+ymat**2 <= radius**2
    full_shape = data_cube[0].shape
    apt_list = {}

    for n in range(len(data_cube)):
        frame = data_cube[n]
        std = noise_cube[n]**2
        
        x0 = int(xc[n])
        y0 = int(yc[n])
        tx = xc[n]-x0
        ty = yc[n]-y0
        
        for i in range(2):
            ind0 = x0 + i
            for j in range(2):
                ind1 = y0 + j
                if (ind0, ind1) not in apt_list:
                    apt_list[(ind0, ind1)] = make_full_apt(full_shape, apt, ind0, ind1)
        # Do integer aperture photometry centered on pixels surrounding
        # the float value centre, and then bi-linearly interpolate.
        ind00 = apt_list[(x0, y0)]
        ind01 = apt_list[(x0, y0+1)]
        ind10 = apt_list[(x0+1, y0)]
        ind11 = apt_list[(x0+1, y0+1)]
        
        flux[n] = ((1-tx) * (1-ty) * np.sum(frame[ind00]) + 
                    tx * (1-ty) * np.sum(frame[ind10]) + 
                    (1-tx) * ty * np.sum(frame[ind01]) + 
                    tx * ty * np.sum(frame[ind11]))
        
        noise[n] = ((1-tx) * (1-ty) * np.sum(std[ind00]**2) + 
                     tx * (1-ty) * np.sum(std[ind10]**2) + 
                     (1-tx) * ty * np.sum(std[ind01]**2) + 
                     tx * ty * np.sum(std[ind11]**2))**0.5
    return flux, noise


def make_full_apt(full_shape, apt, ixc, iyc):
    """Help function to apt_phot_cube, produces an aperture mask of 
    shape full_shape by placing the aperture apt centered on integer pixel
    coordinates according to ixc, iyc.
    """
    full_apt = np.zeros(full_shape, dtype='?')
    radius = int((apt.shape[0]-1)/2)
    full_apt[(iyc-radius):(iyc+radius+1), (ixc-radius):(ixc+radius+1)] = apt
    return full_apt


def sigma_clip(data, clip=5, niter=5):
    """A simple 1D sigma-clipping routine that returns
    a boolean array of indices to non-clipped data    
    """
    ind = np.ones(data.shape, dtype='?')
    for _n in range(niter):
        sigma = np.std(data[ind])
        m = np.median(data[ind])
        ind = np.abs(data-m) < clip*sigma
    return ind


def smo_spl_bg(BJD, BG, smo_len=10, smo_lim=1.2):
    """Smoothing spline fit for background. smo_len is number of data
    points to smooth over.
    """
    ind = BG < smo_lim*np.median(BG)
    tn = np.arange(smo_len, len(BJD[ind])-smo_len, smo_len)
    spl = LSQUnivariateSpline(BJD[ind], BG[ind], BJD[ind][tn])
    BG[ind] = spl(BJD[ind])
    return BG


def load_lc(name, visit, version=0, postfix=''):
    """Load lightcurve, returns dict data structure
    """
    from .config import conf
    filename = os.path.join(conf.data_root,
                            name,
                            visit,
                            'Outdata',
                            '{:05d}'.format(version),
                            f'{name}_{visit}{postfix}.fits')
    return lightcurve(filename)


def load_sa(name, visit, version=0, postfix=''):
    """Load subarray lightcurve, returns dict data structure
    (if it exists)
    """
    return load_lc(name, visit, version, postfix=f'{postfix}_sa')


def load_im(name, visit, version=0, postfix=''):
    """Load imagette lightcurve, returns dict data structure
    (if it exists)
    """
    return load_lc(name, visit, version, postfix=f'{postfix}_im')


def load_binary_sa(name, visit, version=0, postfix=''):
    """Load binary subarray lightcurve, returns dict data structure
    (if it exists)
    """
    return load_sa(name, visit, version, postfix='_binary')


def load_binary_im(name, visit, version=0, postfix=''):
    """Load binary imagette lightcurve, returns dict data structure
    (if it exists)
    """
    return load_im(name, visit, version, postfix='_binary')


def load_drp(name, visit, desc='DEFAULT'):
    """Reads lightcurve extracted by the CHEOPS Data Reduction Pipeline.
    Returns DRP dict, if found.
    """
    from .config import conf
    datapath = os.path.join(conf.data_root, name, visit)
    def find_file(substring):
        for file in os.listdir(datapath):
            if substring in file:
                return os.path.join(datapath, file)
        raise Exception(f'[load_drp] Error: \"{substring}\" not found')

    filename = find_file('SCI_COR_Lightcurve-{:s}'.format(desc))
    return lightcurve(filename)
