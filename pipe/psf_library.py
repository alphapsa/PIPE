# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:37:56 2022

@author: Alexis Brandeker

Handles the PSF library that is a directory structure containing files
defining PSF of various parameters. In the root psf_ref_path folder,
there are one folder per detector position (e.g. 291x830) of the library,
and in each detector position folder there are then a list of files of the
format psf_05690K_18.0K_59323_04.4_5.pkl that defines spline coefficients
for the PSF of given parameters SED Teff (of star used), thermFront_2
(telescope tube temperature), MJD date (integer days), exposure time per
frame of the PSF defining observation (in seconds, for the imagettes if
existing, otherwise subarrays).

The PSF_Library sets up a data base of the existing PSFs and produces
a best match for given target parameters. This match can then be used to
produce an eigen library consisting of the principal components of the
PSF used in the observations.
"""

import os
import numpy as np

# These are the default weights used by the metric, used to define
# how well PSF parameters match. Format:
# (xc, yc, Teff, thermFront_2, MJD, exptime)
DEFAULT_WEIGHTS = (1e2, 1e2, 4e3, 1e4, 0.0, 0.0)

class PSF_Library:
    def __init__(self, psf_ref_path, weights=DEFAULT_WEIGHTS):
        """Initiates the psf library found in the psf_ref_path
        weights are the weights used by the metric
        """
        self.psf_ref_path = psf_ref_path 
        self.serial_limit = 10000 # Largest allowed serial
        self.params, self.files = self.populate_library()
        self.set_metric_weights(weights)
        self.target_x = None
        self.target_y = None
        self.target_Teff = None
        self.target_TF2 = None
        self.target_mjd = None
        self.target_exptime = None

    def get_filename(self, n):
        return os.path.join(self.psf_ref_path, self.files[n])


    def get_params(self, n):
        return self.params[n]


    def populate_library(self):
        """Checks all PSF files in the psf_path, extracts parameters
            from the filenames and adds them to a matrix with columns
            Teff, TF2, exptime, and mjd
            Returns this matrix and a numpy array of filenames
        """
        filenames = []
        
        for folder in os.listdir(self.psf_ref_path):
            folder_path = os.path.join(self.psf_ref_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if os.path.isfile(os.path.join(folder_path, file)):
                        filenames.append(os.path.join(folder, file))
    
        np_filenames = np.array(filenames, dtype=object)
        psf_params = np.zeros((len(np_filenames), 6))
  
        for n, filename in enumerate(filenames):
            psf_params[n] = params_from_filename(filename)
    
        return psf_params, np_filenames


    def set_metric_weights(self, weights):
        """Changes the weights from the default to something custom
        """
        self.metric_weights = weights


    def compute_scores(self, target_params):
        """Computes the PSF distance metric for all entries in psf_params matrix,
        given target parameters. Returns the score vector and an sorting index.
        """
        score = np.zeros(len(self.params))
        for n, psf_entry in enumerate(self.params):
            score[n] = psf_metric(target_params, psf_entry,
                                  weights=self.metric_weights)
        return score


    def best_matches(self, target_params, min_num=5, score_lim=None):
        """Find and return a list of filnames (full path) of at least
        min_num best PSF matches, or all that have a better
        score than score_lim.
        """
        score = self.compute_scores(target_params)
        ind = np.argsort(score)
        sort_filenames = self.files[ind]
        if score_lim is None:
            num = min_num
        else:
            num = max(min_num, np.sum(score<score_lim))
        num = min(num, len(self.files))
        return sort_filenames[:num], score[ind][:num]


    def best_Teff_matches(self, Teff, min_num=5, score_lim=None):
        """Use default values for parameters from target, except for
        Teff, and return best matches. Used for finding suitable PSFs
        for background stars in frame.
        """
        target_params = (self.target_x, self.target_y, Teff, self.target_TF2,
                         self.target_mjd, self.target_exptime)
        return self.best_matches(target_params=target_params,
                                 min_num=min_num, score_lim=score_lim)[0]


    def filename(self, xc, yc, Teff, TF2, mjd, exptime, serial=None, outdir=None):
        """Produces a new unique filename for for a PSF, encoding information
        about the PSF in the filename as following:
        {psf_ref_path}/{xc}x{yc}/psf_{Teff}K_{TF2}K_{mjd}_{exptime}_{serial}.pkl
        where psf_ref_path is the path to the PSF reference data location,
        {xc}x{yc} is a directory encoding the detector position of the PSF
            (created if not existing)
        {Teff} is the effective temperature of the star used to create the PSF,
        {TF2} is the thermFront_2 sensor temperature (negative, to remove the sign),
        {mjd} is the integer MJD
        {exptime} is the exposure time per (coadded) frame (imagette or subarray)
        {serial} is a serial, the smallest number to make the filename unique
            (if not specified)
        """
        if outdir is None:
            outdir = self.psf_ref_path
        dirname = os.path.join(outdir, '{:03d}x{:03d}'.format(xc, yc))
        part1 = 'psf_{:05d}K_{:05.2f}C_{:5.0f}_{:04.1f}'.format(int(Teff), -TF2, mjd, exptime)
    
        os.makedirs(dirname, exist_ok=True)
    
        if serial is None:
            for serial in range(self.serial_limit):
                filename = os.path.join(dirname, part1 + '_{:04d}.npy'.format(serial))
                if not os.path.isfile(filename):
                    break
    
        return os.path.join(dirname, part1 + '_{:04d}.npy'.format(serial))


def params_from_filename(filename):
    """Extracts parameters from filename of the format
     {xc}x{yc}/psf_{Teff}K_{TF2}K_{mjd}_{exptime}_{serial}.npy
     012345678901234567890123456789012345678901234567890123456789
    '291x830/psf_05690K_18.00C_59323_04.4_0005.npy' (as an example)
    """
    xc  = int(filename[:3])
    yc  = int(filename[4:7])
    Teff = int(filename[12:17])
    TF2 = -float(filename[19:24])
    mjd = int(filename[26:31])
    exptime = float(filename[32:36])
    return (xc, yc, Teff, TF2, mjd, exptime)


def psf_diff(target_params, psf_params):
    dxc = target_params[0] - psf_params[0]
    dyc = target_params[1] - psf_params[1]
    dTeff = (target_params[2]/psf_params[2] - 1)
    dTF2 = (target_params[3]/psf_params[3] - 1)
    dmjd = ((target_params[4]-psf_params[4])/1000.0)
    return (dxc, dyc, dTeff, dTF2, dmjd)
    

def psf_metric(target_params, psf_params, weights=DEFAULT_WEIGHTS):
    """Compute a score for the distance between the
    target parameters (xc, yc, Teff, TF2, mjd) and
    PSF parameters (xc, yc, Teff, TF2, mjd, exptime). The
    lower the score, the better the match. The various terms can be 
    customly weighted using the weights parameter. Returns the score.
    """
    weights = np.array(weights)
    weights /= 0.5*np.sum(weights**2)**.5
    
    (dxc, dyc, dTeff, dTF2, dmjd) = psf_diff(target_params, psf_params)
    
    wxc = dxc*weights[0]
    wyc = dyc*weights[1]
    wTeff = dTeff*weights[2]
    wTF2 = dTF2*weights[3] 
    wmjd = dmjd*weights[4]
    wexptime = (psf_params[5]/60)*weights[5]

    return (wxc**2 + wyc**2 + wTeff**2 + wTF2**2 + wmjd**2 + wexptime**2)**.5


