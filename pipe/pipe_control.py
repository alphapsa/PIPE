# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:57:13 2020

@author: Alexis Brandeker, alexis@astro.su.se

Top level routines for using PIPE, to be called by scripts etc.
"""
import os
import pickle
import numpy as np
from .psf_phot import PsfPhot
from .read import lightcurve
from .make_multi_psf import MultiPSFMaker
from .spline_pca import SplinePCA


class PipeControl():
    def __init__(self, pipe_params):
        self.pps = pipe_params    # Set of parameters to be used
        self.pp = None            # PsfPhot object initialised by class
        self.sa_psf_cube0 = None  # Parameter used for binary PSF derivation
        self.sa_psf_cube1 = None  # Parameter used for binary PSF derivation
        self.im_psf_cube0 = None  # Parameter used for binary PSF derivation
        self.im_psf_cube1 = None  # Parameter used for binary PSF derivation

        
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
        used in PSF fit. sigma_clip is the clipping factor to mask bad pixels.
        For poor PSF fits, sigma_clip should be high in order to not mask
        perfectly valid but poorly fit pixels.
        """
        self.process_eigen_sa()
        if self.pps.file_im is not None:
            self.process_eigen_im()


    def process_eigen_sa(self):
        """Process PSF photometry as defined by the pipe_param parameters,
        on subarrays. klip is how many principal components should be
        used in PSF fit. sigma_clip is the clipping factor to mask bad pixels.
        For poor PSF fits, sigma_clip should be high in order to not mask
        perfectly valid but poorly fit pixels.
        """
        if self.pp is None:
            self.pre_proc()
        self.sa_scale, self.sa_bg, self.sa_flux, self.sa_err, self.sa_sel, self.sa_w = \
            self.pp.process_eigen_sa()


    def process_eigen_im(self):
        """Process PSF photometry as defined by the pipe_param parameters,
        on imagettes.  klip is how many principal components should be
        used in PSF fit. sigma_clip is the clipping factor to mask bad pixels.
        For poor PSF fits, sigma_clip should be high in order to not mask
        perfectly valid but poorly fit pixels.
        """
        if self.pp is None:
            self.pre_proc()
        self.im_scale, self.im_bg, self.im_flux, self.im_err, self.im_sel, self.im_w = \
            self.pp.process_eigen_im()
    
    
    def make_psf_lib(self, lib_num=None, klip=None, phase=None, sub_orbits=1):
        """Produces a library of PSFs. The data series is divided into orbits, 
        with one PSF determined per orbit. If phase is defined, the orbits are
        selected to start at phase rather than the first data point. sub_orbits
        can be defined to divide the orbits into smaller data sets, to e.g.
        check if there are systematic PSF differences with roll angle.
        """
        if self.pp is None:
            self.pre_proc()
        if lib_num is None:
            lib_num = self.pp.find_next_lib_num(self.pp.psf_name)
        pm = MultiPSFMaker(self.pp)
        sa_ranges = pm.find_ranges(phase=phase, sub_orbits=sub_orbits)
        if klip is None:
            klip = len(sa_ranges)
        return pm.prod_psf(sa_ranges[:klip], lib_num)
    
    
    def combine_psf_lists(self, lib_num_list):
        """From a list of library numbers, this method reads the
        corresponding PSF library files and merges them, with the
        merged list of PSFs returned. This list can then be used
        to derive an eigen library of principal PSF components.
        """
        if self.pp is None:
            self.pre_proc(pproc=False)

        psf_lib = []
        for lib_num in lib_num_list:
            file_name = self.pp.psf_name(lib_num)
            with open(file_name, 'rb') as fp:
                psf_lib.extend(pickle.load(fp))
        return psf_lib
    
    
    def make_eigen(self, psf_lib, out_num):
        """From a list of PSFs, produce a list of principal component
        functions and save in a file ("eigen library"), enumerated by out_num.
        The eigen library list is also returned.
        """
        if self.pp is None:
            self.pre_proc(pproc=False)
        sp = SplinePCA(psf_lib)
        eigen_lib = sp.get_eigen_spline_lib()
        filename = self.pp.eigen_name(out_num)
        with open(filename, 'wb') as fp:
            pickle.dump(eigen_lib, fp)        
        self.pp.mess('--- {:d} PSF eigen functions produced, saved to \'{:s}\''.format(
                len(eigen_lib), filename))
        return eigen_lib    


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
        if self.pp is None:
            self.pre_binary()
        self.sa_psf_cube0, self.sa_psf_cube1, self.sa_bg = \
            self.pp.process_binary_sa()
        if self.pps.file_im is not None:
            self.im_psf_cube0, self.im_psf_cube1, self.im_bg = \
                self.pp.process_binary_im()

        
    def make_binary_psfs(self, lib_num, klip=None, phase=None, sub_orbits=1):
        """Produces one library of PSFs for each component of a binary. The
        data series is divided into orbits, with one PSF determined per orbit.
        if phase is defined, the orbits are selected to start at phase rather
        than the first data point. sub_orbits can be defined to divide the
        orbits into smaller data sets, to e.g. check if there are systematic 
        PSF differences with roll angle.
        """
        # An implementation trick used to derive PSFs for the primary and
        # secondary separately, is to produce PSF models for both
        # components and first subtract the secondary from the data
        # cube, derive the primary PSF, and vice versa subtract the
        # primary from the data cube to derive the secondary PSF.
        # To have a good initial PSF estimate for the first iteration 
        # is therefore important for proper subtraction without inducing
        # too much PSF noise.
        
        if self.sa_psf_cube0 is None:
            self.process_binary()
        pm = MultiPSFMaker(self.pp)
        sa_ranges = pm.find_ranges(phase=phase, sub_orbits=sub_orbits)
        if klip is None:
            klip = len(sa_ranges)
        
        def add_noise_sa(mc):
            if self.pps.bgstars:
                mc += 2*self.pp.sa_bgstars
            if self.pps.smear_corr:
                mc += np.abs(self.pp.sa_smear[:, None, :])
            if self.pps.darksub:
                mc += self.pp.sa_dark
            if self.pps.remove_static:
                mc += np.abs(self.pp.sa_stat_res)
            return mc

        def add_noise_im(mc):
            if self.pps.bgstars:
                mc += 2*self.pp.im_bgstars
            if self.pps.smear_corr:
                mc += np.abs(self.pp.im_smear[:, None, :])
            if self.pps.darksub:
                mc += self.pp.im_dark
            if self.pps.remove_static:
                mc += np.abs(self.pp.im_stat_res)
            return mc
        
        # Prepare for producing PSF of primary
        self.pp.sa_xc = self.pp.sa_xc0
        self.pp.sa_yc = self.pp.sa_yc0
        model_cube = (np.abs(self.sa_psf_cube0) + 2*np.abs(self.sa_psf_cube1) +
                      self.sa_bg[:, None, None])
        self.pp.sa_noise = self.pp.psf_noise_sa(add_noise_sa(model_cube))
        self.pp.sa_sub -= self.sa_psf_cube1
    
        if self.pps.file_im is not None:
            self.pp.im_xc, self.pp.im_yc = self.pp.im_xc0, self.pp.im_yc0
            model_cube = (np.abs(self.im_psf_cube0) + 2*np.abs(self.im_psf_cube1) +
                          self.im_bg[:, None, None])
            self.pp.im_noise = self.pp.psf_noise_im(add_noise_sa(model_cube))
            self.pp.im_sub -= self.im_psf_cube1
            
        pm = MultiPSFMaker(self.pp)
        # Produce primary PSF list
        pm.prod_psf(sa_ranges[:klip], lib_num=lib_num)

        # Prepare for producing PSF of secondary
        self.pp.sa_sub += self.sa_psf_cube1
        self.pp.sa_xc, self.pp.sa_yc = self.pp.sa_xc1, self.pp.sa_yc1
        model_cube = (2*np.abs(self.sa_psf_cube0) + np.abs(self.sa_psf_cube1) +
                      self.sa_bg[:, None, None])
        self.pp.sa_noise = self.pp.psf_noise_sa(add_noise_sa(model_cube))
        self.pp.sa_sub -= self.sa_psf_cube0

        pm = MultiPSFMaker(self.pp)
        # Don't use imagettes for secondary
        pm.im = False

        # Produce secondary PSF list
        pm.prod_psf(sa_ranges[:klip], lib_num=lib_num)

        # Restate
        self.pp.sa_xc, self.pp.sa_yc = self.pp.sa_xc0, self.pp.sa_yc0
        self.pp.sa_sub += self.sa_psf_cube0
        model_cube = (np.abs(self.sa_psf_cube0) + np.abs(self.sa_psf_cube1) +
                      self.sa_bg[:, None, None])
        self.pp.sa_noise = self.pp.psf_noise_sa(add_noise_sa(model_cube))

        if self.pps.file_im is not None:
            self.pp.im_sub += self.im_psf_cube1
            model_cube = (np.abs(self.im_psf_cube0) + np.abs(self.im_psf_cube1) +
                          self.im_bg[:, None, None])
            self.pp.im_noise = self.pp.psf_noise_im(add_noise_im(model_cube))


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


