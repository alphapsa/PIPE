# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:57:13 2020

@author: Alexis Brandeker, alexis@astro.su.se

Top level routines for using PIPE, to be called by scripts etc.
"""
import os
import numpy as np
from .psf_phot import PsfPhot
from .read import lightcurve
from .make_multi_psf import MultiPSFMaker
from .spline_pca import SplinePCA


class PipeControl():
    def __init__(self, pipe_params):
        self.pps = pipe_params    # Set of parameters to be used
        self.pp = None            # PsfPhot object initialised by class

        
    def pre_proc(self, pproc=True):
        """Initialise PsfPhot class and do pre-processing on data (i.e. basic 
        processing before photometric PSF extraction). If pproc is False, the
        pre-processing step is skipped.
        """
        os.makedirs(self.pps.outdir, exist_ok=True)
        self.pp = PsfPhot(self.pps)
        self.pp.mess(f'--- {self.pps.name}/{self.pps.visit} (version {self.pps.version})')
        if pproc: self.pp.pre_process()

    
    def process_eigen(self):
        """Process PSF photometry as defined by the pipe_param parameters,
        on both subarrays and imagettes (if they exist).
        klip is how many principal components should be
        used in PSF fit.
        """
        self.process_eigen_sa()
        if self.pps.file_im is not None:
            self.process_eigen_im()


    def process_eigen_sa(self):
        """Process PSF photometry as defined by the pipe_param parameters,
        on subarrays. klip is how many principal components should be
        used in PSF fit.
        """
        if self.pp is None:
            self.pre_proc()
        self.sa_scale, self.sa_bg, self.sa_flux, self.sa_err, self.sa_sel, self.sa_w = \
            self.pp.process_eigen_sa()


    def process_eigen_im(self):
        """Process PSF photometry as defined by the pipe_param parameters,
        on imagettes.  klip is how many principal components should be
        used in PSF fit.
        """
        if self.pp is None:
            self.pre_proc()
        self.im_scale, self.im_bg, self.im_flux, self.im_err, self.im_sel, self.im_w = \
            self.pp.process_eigen_im()
    
    
    def make_psf_lib(self, outdir=None):
        """Produces a library of PSFs. The data series is divided into orbits, 
        with one PSF determined per orbit.
        """
        if self.pp is None:
            self.process_eigen()
        pm = MultiPSFMaker(self.pp, max_threads=self.pps.nthreads)
        sa_ranges = pm.find_ranges()
        return pm.prod_psf(sa_ranges, outdir=outdir)
    
    
    def load_drp(self, desc='DEFAULT'):
        """Reads lightcurve extracted by the CHEOPS Data Reduction Pipeline.
        Returns DRP dict.
        """
        if desc == 'DEFAULT':
            return lightcurve(self.pps.file_lc_default)
        if desc == 'OPTIMAL':
            return lightcurve(self.pps.file_lc_optimal)
        if desc == 'RINF':
            return lightcurve(self.pps.file_lc_rinf)
        if desc == 'RSUP':
            return lightcurve(self.pps.file_lc_rsup)
        raise Exception(f'[load_drp] Error: {desc} not defined')


    def load_lc(self, postfix='_sa'):
        """Load lightcurve, returns dict data structure (if it exists)
        """
        filename = os.path.join(self.pps.outdir,
                                f'{self.pps.name}_{self.pps.visit}{postfix}.fits')
        return lightcurve(filename)

    def load_sa(self):
        """Load subarray lightcurve, returns dict data structure
        (if it exists)
        """
        return self.load_lc(postfix='_sa')


    def load_im(self):
        """Load imagette lightcurve, returns dict data structure
        (if it exists)
        """
        return self.load_lc(postfix='_im')


    #----------- Methods for binary extractions below

    def pre_binary(self):
        """Initialise PsfPhot class and do pre-processing on data (i.e. basic 
        processing before photometric PSF extraction).
        """
        self.pps.binary = True
        os.makedirs(self.pps.outdir, exist_ok=True)    
        self.pp = PsfPhot(self.pps)
        self.pp.mess(f'--- Binary {self.pps.name}/{self.pps.visit} (version {self.pps.version})')
        self.pp.pre_binary()


    def process_binary(self):
        """Extract photometry of a binary from the subarrays and the imagettes,
        if they exist. Saves PSF models for the two components to class
        variables. 
        """
        self.pps.binary = True
        if self.pp is None:
            self.pre_binary()
        self.sa_psf_cube0, self.sa_psf_cube1, self.sa_bg = \
            self.pp.process_binary_sa()
        if self.pps.file_im is not None:
            self.im_psf_cube0, self.im_psf_cube1, self.im_bg = \
                self.pp.process_binary_im()

  
    def load_binary_sa(self):
        """Load binary subarray lightcurve, returns dict data structure
        (if it exists)
        """
        return self.load_lc(postfix='_binary_sa')


    def load_binary_im(self):
        """Load binary imagette lightcurve, returns dict data structure
        (if it exists)
        """
        return self.load_lc(postfix='_binary_im')


