# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:27:24 2020

@author: Alexis Brandeker, alexis@astro.su.se

Master class PsfPhot of PIPE contains the data and routines to
manipulate and extract PSF photometry. Has separate but 
similar paths for subarray [sa] and imagette [im] manipulation,
and also for entangled binaries where two componens are fit 
simultaneously.

Most of the detailed functionality is provided by other modules
that are called from PsfPhot.

"""
import os
import pickle
import numpy as np
from astropy.io import fits

from .analyse import psf_phot_cube
from .cent import (
    flux as cent_flux
)
from .multi_cent import (
    deconvolve as multi_cent_deconvolve,
    binary_deconvolve as multi_cent_binary_deconvolve,
    binary_psf as multi_cent_binary_psf,
    binary_psf_fix as multi_cent_binary_psf_fix
)
from .multi_level import estimate as level_estimate
from .level import ring_median as level_ring_median
from .multi_psf import fit as multi_psf_fit, fit_binary as multi_psf_fit_binary
from .optimal_params import TestParams, FindOptimal
from .pipe_log import PipeLog
from .pipe_statistics import mad, sigma_clip
from .psf_model import psf_model
from .psf_library import PSF_Library
from .read import (
    imagette_offset, raw_datacube as read_raw_datacube, attitude, gain as read_gain,
    bias_ron_adu, thermFront_2, mjd2bjd, nonlinear, flatfield, starcat,
    save_eigen_fits, save_bg_star_phot_fits, save_binary_eigen_fits, sub_image_indices,
    dark as read_dark, bad as read_bad, PSFs as load_PSFs, read_psf_filenames,
    save_psf_filenames, save_cube_fits
)
from .spline_pca import SplinePCA
from .syntstar import star_bg, rotate_position, derotate_position, psf_radii
from .multi_star import make_star_bg, refine_star_bg, make_bg_circ_mask_cube, make_bg_psf_mask_cube
from .reduce import (
    resample_attitude, resample_imagette_time, aperture, integrate_psf,
    interp_cube_ext, cube_apt, clean_cube2D, interp_cube, noise, psf_noise,
    pix_mat, make_maskcube, rough_contrast, check_low, check_val, check_pos,
    check_motion, check_mask, empiric_noise, cti_corr_fun, resid_smear
)
from .version import __version__


class PsfPhot:
    """ Reads and stores CHEOPS data and collects methods to analyse
        the data using mainly PSF photometry.
    """
    def __init__(self, pipe_params):
        self.pps = pipe_params
        if self.pps.sa_optimise or self.pps.im_optimise:
            plog_verbosity = 0
        else:
            plog_verbosity = self.pps.plog_verbosity
        self.plog = PipeLog(self.pps.file_log, plog_verbosity)
        self.mess = self.plog.mess  # Shorthand
        
        self.mess(f'PIPE version {__version__}')
        self.plog.mess_list(self.pps.str_list()) # Save list of params to log
        
        if self.pps.file_starcat is None:
            self.pps.bgstars = False
            self.pps.fit_bgstars = False

        if self.pps.Teff is None:
            if self.pps.file_starcat is None:
                raise Exception('Without starcat, Teff needs to be defined')
            self.pps.Teff = self.read_Teff()

        # If BLAS is multithreaded, reduce the number of threads used on top level
        self.nthreads_reduced =  int(max(round(self.pps.nthreads/self.pps.nthreads_blas), 1))


        # ----------- General variables
        self.eigen_psf = None   # Library of PSF eigen components
        self.psf = None         # Default PSF to be used when not
                                # fitting eigen components
        self.starcat = None     # Star Catalogue object containing bg stars 
        self.mjd2bjd = None     # Defines a function to be used to convert
                                # between MJD and BJD
        self.cti_corr_fun = None # The CTI correction function, weakly epoch dependent  
        # ----------- Subarray variables
        self.sa_debias = None   # Cube of DRP calibrated subarray frames
                                # (bias removed, flatfielded, nonlinear corrected)
        self.sa_sub = None      # Reduced cube with background (and dark)
                                # subtracted, smearing corrected
        self.sa_bgstars = None  # Cube of background stars in subarray
        self.sa_bg_mask = None  # Cube of PSF core mask of BG stars
        self.sa_dark = None     # Dark current subarray image scaled to exp. time
        self.sa_dark_err = None # Std error of subarray dark current image
        self.sa_mask = None     # Mask image for bad pixels in subarray
        self.sa_apt = None      # Image that defines active pixels in cube
                                # (e.g. with circular aperture not all pixels)
        self.sa_bg = None       # Background value, a constant for each frame
                                # in cube list
        self.sa_dbg = 0         # Differential background value, used to correct bg
        self.sa_xc = None       # Array of PSF x-coordinate centres for subarray
        self.sa_yc = None       # Array of PSF y-coordinate centres for subarray
        self.sa_mjd = None      # Array of MJD epochs for subarray
        self.sa_hdr = None      # Calibrated subarray cube fits-header
        self.sa_tab = None      # Calibrated subarray cube table
        self.sa_att = None      # Attitude array data for subarray frames
                                # 0 = MJD_TIME, 1 = SC_RA, 2 = SC_DEC,
                                # 3 = SC_ROLL_ANGLE
        self.sa_t = None        # Interp parameter (norm epochs from 0 to 1)
        self.sa_smear = None    # Smearing correction, 1D vector for each frame
        self.sa_stat_res = None # Median of residuals, a derived static image
                                # Can be used as a dark current image
        self.sa_thermFront_2 = None  # thermFront_2 vector, passed on to save-file
        self.sa_nexp = None     # Number of exposures per subarray frame
        self.sa_flux = None     # Latest flux estimate of frames
        self.sa_sel = None      # Selection of 'good' frames (flag==0)
        self.sa_norm = None     # Median flux of frames, used for background star scaling

        # ----------- Imagette variables
        self.im_nexp = None     # Number of exposures per imagette frame
        self.nexp = None        # subarray/imagette relative exposure time
        self.gain_fun = None    # Gain in e/ADU as a function of MJD
        self.im_off = None      # Offset of imagette relative full frame
        self.im_sa_off = None   # Offset of imagette relative subarray
        self.im_raw = None      # Raw data cube of imagettes
        self.im_dark = None     # Dark imagette image scaled to exposure time
        self.im_dark_err = None # Std error of imagette dark current image
        self.im_mask = None     # Mask image for bad pixels in image
        self.im_mjd = None      # Array of MJD epochs for imagettes
        self.im_hdr = None      # Raw imagette cube fits-header
        self.im_tab = None      # Raw imagette cube table
        self.im_att = None      # Attitude array data for imagettes
                                # 0 = MJD_TIME, 1 = SC_RA, 2 = SC_DEC,
                                # 3 = SC_ROLL_ANGLE
        self.im_t = None        # Interp parameter (norm epochs from 0 to 1)
        self.im_apt = None      # Image that defines active pixels in cube
        self.im_bgstars = None  # Cube of background stars in imagettes
        self.im_bg_mask = None  # Cube of PSF core mask of BG stars
        self.im_bg = None       # Background value, a constant for each frame
                                # in cube list
        self.im_dbg = 0         # Differential background value, used to correct bg
        self.im_xc = None       # Array of PSF x-coordinate centres for imagettes
        self.im_yc = None       # Array of PSF y-coordinate centres for imagettes
        self.im_smear = None    # Smearing correction, 1D vector for each frame
        self.im_stat_res = None # Median of residuals, a derived static image
                                # Can be used as a dark current image
        self.im_thermFront_2 = None  # thermFront_2 vector, passed on to save-file
        self.im_flux = None     # Latest flux estimate of imagettes
        self.im_sel = None      # Selection of 'good' frames (flag==0)
        self.im_norm = None     # Median flux of imagette, used for background star scaling

        self.sa_mad = None
        self.im_mad = None

        # Read and initialise data from files
        self.read_data()


    def read_data(self):
        """Initialises data from data files
        """
        # Define gain, bias, and read-out noise        
        self.make_gain_function()
        self.read_bias_ron()

        # Read subarray data cube and define
        # sa_debias, sa_mjd, sa_hdr, sa_tab
        self.read_subarrays()
        
        # Check if imagettes are available, load data and define
        # corresponding parameters
        if self.pps.file_im is not None:
            self.read_imagettes()
            # A bug early on (<2021) in the DPS caused imagette timings to be wrong.
            # This is fixed by resampling times from subarray timings.
            if self.pps.resample_im_times:
                self.im_mjd = resample_imagette_time(self.sa_mjd, self.nexp)

        self.read_attitude()
        self.read_thermFront_2()
        self.read_mask()
        self.read_darks()
        self.make_mjd2bjd_function()
        self.define_psf_library()
        self.read_starcat()


    def define_psf_library(self):
        """PCA decomposes PSF files from psf_files list. If none is defined, 
        best matches from PSF library are found that match the parameters of
        the target in terms of position on detector, effective temperature of 
        target SED, proximity in time (MJD), exposure time, and range of 
        thermFront_2 sensor values during visit.
        """

        if self.pps.psf_filenames_file is None:
            psf_files = None
        else:
            psf_files = read_psf_filenames(self.pps.psf_filenames_file)

        self.psf_lib = PSF_Library(os.path.join(self.pps.calibpath, 'psf_lib'))

        ixoff = self.sa_hdr['X_WINOFF'] + int(self.sa_debias[0].shape[1]/2)
        iyoff = self.sa_hdr['Y_WINOFF'] + int(self.sa_debias[0].shape[0]/2)
        Teff = self.pps.Teff
        TF2_min = np.percentile(self.sa_thermFront_2, 5)
        TF2_med = np.median(self.sa_thermFront_2)
        TF2_max = np.percentile(self.sa_thermFront_2, 95)
        mjd = np.nanmedian(self.sa_mjd)
        # Current parameters
        if self.pps.file_im is not None:
            exptime = self.im_hdr['EXPTIME'] * self.im_nexp
        else:
            exptime = self.sa_hdr['TEXPTIME']

        self.psf_lib.target_x = ixoff
        self.psf_lib.target_y = iyoff
        self.psf_lib.target_Teff = Teff
        self.psf_lib.target_TF2 = TF2_med
        self.psf_lib.target_mjd = mjd
        self.psf_lib.target_exptime = exptime

        if psf_files is None:
            self.mess('PSF parameters: (x, y) = ({:03d},{:03d})'.format(ixoff, iyoff))
            self.mess('Teff = {:.0f}K, MJD = {:.1f}, exptime = {:04.1f}'.format(
                Teff, mjd, exptime))
            self.mess('Range of TF2: (min, med, max) = ({:04.1f}C, {:04.1f}C, {:04.1f}C)'.format(
                TF2_min, TF2_med, TF2_max))

            psf_files = []
            for TF2 in [TF2_min, TF2_med, TF2_max]:
                more_files, scores = self.psf_lib.best_matches((ixoff, iyoff, Teff, TF2, mjd, exptime),
                                                                min_num=self.pps.psf_min_num,
                                                                score_lim=self.pps.psf_score)
                psf_files = np.concatenate((psf_files, more_files))
                self.mess('For TF2 = {:.2f}C: {:d} PSFs, score {:.2f} to {:.2f}'.format(
                          TF2, len(more_files), scores[0], scores[-1]))
            psf_files = np.unique(psf_files)
        else:
            self.mess('Using pre-defined PSF list')
            
        self.mess('Number of unique PSFs used: {:d}'.format(len(psf_files)))
        if len(psf_files) < 10:
            self.mess('WARNING: number of PSFs less than 10')

        # Read PSF files and PCA decompose
        psflist = load_PSFs(psf_files, self.psf_lib.psf_ref_path)
        num_eigen = min(max(self.pps.klip, 10), len(psf_files))

        if self.pps.save_psf_list:
            save_psf_filenames(os.path.join(self.pps.outdir, 'psf_filenames.txt'), psf_files)

        self.mess('PCA decomposing PSFs into {:d} components'.format(num_eigen))
        spca = SplinePCA(psflist, num_eigen=num_eigen, radius=self.pps.normrad)
        eigen_splines = spca.get_eigen_spline_lib()
        self.eigen_psf = [psf_model(psf_spl) for psf_spl in eigen_splines]
        self.psf = self.eigen_psf[0]
        if self.pps.centre_psf_filename is not None:
            self.centre_psf = psf_model(load_PSFs([self.pps.centre_psf_filename], self.psf_lib.psf_ref_path)[0])
        else:
            self.centre_psf = self.psf

        if self.pps.save_psf_pc:
            x = np.linspace(-100, 100, 201)
            pc_cube = np.zeros((len(self.eigen_psf), len(x), len(x)))
            for n in range(len(pc_cube)):
                pc_cube[n] = self.eigen_psf[n](x, x, circular=True)
            self.save_cube_fits('psf_pcs.fits', pc_cube)


    def pre_process(self):
        """Pre-processing reduces the data (without extracting photometry).
        It subtracts the background, removes potential smearing, and also uses
        the PSF to compute accurate centers (through deconvolution).
        """
        self.mess('--- Start pre-processing')
        self.reduce_data_sa()
        
        # Define PSF normalised to subarray flux
        self.mess('Clean frames for de-convolution and centering', 2)
        clean = clean_cube2D(self.sa_sub, self.sa_mask, self.sa_apt)

        if self.pps.centre:
            self.mess('Compute PSF deconv centers [sa] (multi {:d} threads)'.format(self.pps.nthreads))
            # Extract central region to improve deconvolution speed as we only
            # need to consider the central part for centroiding.
            fitrad = min(2*self.pps.centfit_rad, 0.5*np.min(clean.shape[1:]))
            startX = int(clean.shape[2]*0.5-fitrad)
            endX = int(clean.shape[2]*0.5+fitrad)
            startY = int(clean.shape[1]*0.5-fitrad)
            endY = int(clean.shape[1]*0.5+fitrad)
            subcube = clean[:,startY:endY,startX:endX]
            subcube = np.nan_to_num(subcube, copy=False)
            xc, yc = self.cent_deconv(subcube[self.sa_cent_sel])
            xc += startX
            yc += startY
            ind = sigma_clip(xc) * sigma_clip(yc)
            self.mess('SA cent std: ({:.3f}, {:.3f})'.format(np.std(xc[ind]),
                                                        np.std(yc[ind])))
            self.sa_xc[self.sa_cent_sel], self.sa_yc[self.sa_cent_sel] = xc, yc
        
        if self.pps.binary:
            skip_bg_stars = [0, self.pps.secondary]
        else:
            skip_bg_stars = [0]
        
        # Reduce imagettes, if they exist
        if self.pps.file_im is not None:
            self.reduce_data_im()
            self.im_mask_cube = np.ones(self.im_debias.shape, dtype='?')
            self.im_mask_cube[:] = self.im_mask

            if self.pps.centre:
                self.mess('Compute PSF deconv centers [im] (multi {:d} threads)'.format(self.pps.nthreads))
                clean = np.nan_to_num(clean_cube2D(self.im_sub, self.im_mask, self.im_apt))
                xc, yc = self.cent_deconv(clean[self.im_cent_sel])
                ind = sigma_clip(xc) * sigma_clip(yc)
                self.mess('IM cent std: ({:.3f}, {:.3f})'.format(np.std(xc[ind]),
                                                            np.std(yc[ind])))
                self.im_xc[self.im_cent_sel], self.im_yc[self.im_cent_sel] = xc, yc

        self.define_workcat()

        # Compute background stars and smearing
        self.make_star_bg_cube_sa(skip=skip_bg_stars)
        self.make_bg_mask_sa(skip=skip_bg_stars)
        self.compute_smearing_sa()

        self.sa_norm, self.sa_apt_flux = self.comp_fix_apt_phot(self.sa_sub,
                            self.sa_bgstars, self.sa_smear,
                            self.sa_xc, self.sa_yc,
                            self.psf, self.pps.normrad)
        self.sa_flux = self.sa_norm*np.ones(len(self.sa_debias))
        self.sa_sel = np.ones(len(self.sa_flux), dtype='?')

        if self.pps.file_im is not None:
            self.make_star_bg_cube_im(skip=skip_bg_stars)
            self.make_bg_mask_im(skip=skip_bg_stars)
            self.compute_smearing_im()
            self.im_norm, self.im_apt_flux = self.comp_fix_apt_phot(self.im_sub,
                                self.im_bgstars, self.im_smear,
                                self.im_xc, self.im_yc,
                                self.psf, self.pps.normrad)
            self.im_flux = self.im_norm*np.ones(len(self.im_debias))
            self.im_sel = np.ones(len(self.im_flux), dtype='?')

        
    def comp_fix_apt_phot(self, datacube, bg_cube, smear, xc, yc, psf_mod, radius):
        """Extract photometry using a fixed, centred aperture of radius. This
        function corrects for flux losses due to PSF jitter, as well as background
        and smearing contributions. Returns an estimate of the median flux of
        the target, and its variability with frame number
        """
        side = 2*radius
        bg_flux = self.apt_phot(bg_cube, radius=radius)
        data_flux = self.apt_phot(datacube, radius=radius)

        psf_flux = np.zeros(len(datacube))
        x = radius + xc - int(datacube.shape[2]*0.5)
        y = radius + yc - int(datacube.shape[1]*0.5)
        apt = aperture((side,side), radius=radius)

        i0 = int(datacube.shape[2]*0.5)-radius
        v = np.arange(side)
        for n in range(len(datacube)):
            psf_frame = psf_mod(v-x[n], v-y[n])
            if i0 > 0:
                psf_frame += smear[n][None,i0:(i0+side)]
            psf_flux[n] = np.sum(psf_frame*apt)

        median_flux = np.median(data_flux / (bg_flux + psf_flux))
        target_flux = (data_flux - median_flux*bg_flux)/psf_flux

        return median_flux, target_flux


    def bg_model_sa(self):
        """Computes the background model including background stars, 
            smear trails, static image and background adjustment
        """
        bg_mod = np.zeros_like(self.sa_sub)

        if np.isscalar(self.sa_dbg) is False:
            bg_mod += self.sa_dbg[:,None,None]

        if self.pps.bgstars:
            bg_mod += self.sa_norm * self.sa_bgstars
        
        if self.pps.smear_corr:
            bg_mod += self.sa_norm * (self.sa_smear+self.sa_smear_resid)[:, None, :]

        if self.pps.remove_static:
            bg_mod += self.sa_stat_res

        return bg_mod*self.sa_apt


    def bg_model_im(self):
        """Computes the background model including background stars, 
            smear trails, static image and background adjustment
        """
        bg_mod = np.zeros_like(self.im_sub)

        if np.isscalar(self.im_dbg) is False:
            bg_mod += self.im_dbg[:,None,None]

        if self.pps.bgstars:
            bg_mod += self.im_norm * self.im_bgstars
        
        if self.pps.smear_corr:
            bg_mod += self.im_norm * (self.im_smear+self.im_smear_resid)[:, None, :]

        if self.pps.remove_static:
            bg_mod += self.im_stat_res

        return bg_mod*self.im_apt


    def process_eigen_sa(self):
        """Extract photometry from all subarrays by producing best-fit
        PSFs from the PSF eigen library. The return value are
        scale - fitted scale of the PSF
        bg - fitted background
        flux - computed flux from noise-weighted integration
        err - std error of flux estimated from noise in images
        sel - index of frames with source
        w - array of PSF PC weights
        'scale' is typically a more robust estimate of flux than the
        integrated 'flux'.
        """
        self.mess('--- Start processing subarray data with eigen')
        self.sa_bg_refined = False
        max_klip = len(self.eigen_psf)
        self.sa_best_mad = np.inf

        def test_iter(params, niter=self.pps.sigma_clip_niter):
            # Reset some state variables that are modified during iteration
            sel = self.sa_cent_sel
            dbg = np.zeros_like(self.sa_bg)
            self.sa_dbg = np.zeros_like(self.sa_bg)
            self.pps.remove_static = params.bStat
            self.sa_stat_res *= 0
            self.sa_smear_resid *= 0
            self.sa_noise = self.raw_noise_sa()
            self.sa_mask_cube[:] = self.sa_mask
            if params.bBG:
                bg_fit=0
            else:
                bg_fit=-1

            for n in range(niter):
                self.mess('--- Iteration sa {:d}/{:d}'.format(n+1, niter))
                bg_mod = self.bg_model_sa()
                if self.pps.mask_bg_stars:
                    mask_cube = self.sa_mask_cube[sel] * self.sa_bg_mask[sel]
                else:
                    mask_cube = self.sa_mask_cube[sel]
                psf_cube0, scale0, dbg0, w0 = multi_psf_fit(
                                self.eigen_psf[:params.klip],
                                self.sa_sub[sel] - bg_mod[sel],
                                self.sa_noise[sel],
                                mask_cube,
                                self.sa_xc[sel], self.sa_yc[sel],
                                fitrad=params.fitrad,
                                defrad=self.pps.psf_rad,
                                bg_fit=bg_fit,
                                nthreads=self.nthreads_reduced,
                                non_negative=self.pps.non_neg_lsq)
                # Interpolate over frames without source
                t0 = self.sa_att[sel, 0]
                t = self.sa_att[:, 0]
                self.sa_psf_cube = interp_cube_ext(t, t0, psf_cube0*self.sa_apt) 
                w = interp_cube_ext(t, t0, w0)
                scale = np.interp(t, t0, scale0)
                dbg += np.interp(t, t0, dbg0)
                self.sa_dbg = dbg
                    
                self.mess('Iter {:d} MAD sa: {:.2f} ppm'.format(n+1, mad(scale0)))
                self.make_mask_cube_sa()
                self.sa_mask_cube[sel==0] = self.sa_mask
                res = self.compute_residuals_sa()
        
                if self.pps.smear_resid_sa:
                    self.update_smear_sa(res)
                    res = self.compute_residuals_sa()
                if params.bStat:
                    self.compute_resid_stat_sa(res)
                    
                self.sa_noise = (self.psf_noise_sa(np.abs(self.sa_psf_cube) +
                                                np.abs(self.bg_model_sa()) +
                                                np.abs(self.sa_dark))**2 +
                                self.sa_dark_err**2)**.5

                if self.pps.bgstars and self.pps.fit_bgstars and self.sa_bg_refined is False:
                    # Only refine once
                    self.refine_star_bg_sa()
                    self.make_star_bg_cube_sa()
                    self.sa_bg_refined = True

            flux, err = self.psf_phot_sa(self.sa_psf_cube, self.pps.fitrad)
            flag = self.flag_sa(scale, sel)

            sel = (flag==0)
            self.sa_mad = mad(scale[sel])
            self.mess('MAD sa: {:.2f} ppm'.format(mad(scale)))
            curr_mad = mad(scale[sel])
            self.mess('MAD sa[flag==0]: {:.2f} ppm'.format(curr_mad))
            self.sa_flux = scale
            self.sa_sel = sel

            if curr_mad < self.sa_best_mad:
                self.sa_best_mad = curr_mad
                self.save_results_sa(scale, err, w, flag)

            return scale, dbg, flux, err, sel, w

        klip = min(self.pps.klip, max_klip)
        nominal = TestParams(klip=klip, fitrad=self.pps.fitrad, bBG=(self.pps.bg_fit==0),
                             bDark=self.pps.darksub, bStat=self.pps.remove_static)

        if self.pps.sa_optimise is False:
            scale, dbg, flux, err, sel, w = test_iter(nominal)
        else:
            self.mess('Start optimising {:s}/{:s} ({:05d}) [sa]'.format(
                self.pps.name, self.pps.visit, self.pps.version), 0)            
            fo = FindOptimal(self.pps.sa_test_klips, self.pps.sa_test_fitrads,
                             self.pps.sa_test_BG, self.pps.sa_test_Dark, self.pps.sa_test_Stat)
            for n in range(self.pps.optimise_restarts):
                nominal.mad = None
                current_mad = None
                fo.start(nominal, self.pps.optimise_tree_iter)
                while True:
                    testparams = fo.next_test(current_mad)
                    if testparams is None:
                        break
                    if self.pps.sa_test_Dark:
                        self.pps.darksub = testparams.bDark
                        self.reduce_data_sa()
                    scale, dbg, flux, err, sel, w = test_iter(testparams)
                    current_mad = self.sa_mad
                    self.mess('{:s}, mad={:.2f} [sa]'.format(str(testparams), current_mad), 0)
                nominal = fo.get_best()
                self.mess('--- Iter best {:d}: {:s} [sa]'.format(n, str(nominal)), 0)
            self.mess('Optimisation done ({:d} iterations) [sa]'.format(len(fo.tested_params)), 0)
            self.mess(fo.str_tested(), 0)
            self.mess('--- Generating best result [sa]', 0)
            nominal.mad = None
            scale, dbg, flux, err, sel, w = test_iter(nominal, self.pps.sigma_clip_niter+1)
            self.mess('{:s}, mad={:.2f} [sa]'.format(str(nominal), self.sa_mad), 0)
            self.mess('--- Done! [sa]', 0)

        return  scale, dbg, flux, err, sel, w


    def process_eigen_im(self):
        """Extract photometry from all imagettes by producing best-fit
        PSFs from the PSF eigen library. The return value are
        scale - fitted scale of the PSF
        bg - fitted background
        flux - computed flux from noise-weighted integration
        err - std error of flux estimated from noise in images
        sel - index of frames with source
        w - array of PSF PC weights
        'scale' is typically a more robust estimate of flux than the
        integrated 'flux'.
        """
        self.mess('--- Start processing imagette data with eigen')
        self.im_bg_refined = False
        max_klip = len(self.eigen_psf)
        self.im_best_mad = np.inf

        klip = self.pps.klip
        if klip is None:
            klip = len(self.eigen_psf)
        else:
            klip = min(klip, len(self.eigen_psf))


        def test_iter(params, niter=self.pps.sigma_clip_niter):
            # Reset some state variables that are modified during iteration
            sel = self.im_cent_sel
            dbg = np.zeros_like(self.im_bg)
            self.im_dbg = np.zeros_like(self.im_bg)
            self.pps.remove_static = params.bStat
            self.im_stat_res *= 0
            self.im_smear_resid *= 0
            self.im_noise = self.raw_noise_im()
            self.im_mask_cube[:] = self.im_mask
            if params.bBG:
                bg_fit=0
            else:
                bg_fit=-1

            for n in range(niter):
                self.mess('--- Iteration im {:d}/{:d}'.format(n+1, niter))
                # Only extract photometry from frames with source
                bg_mod = self.bg_model_im()
                if self.pps.mask_bg_stars:
                    mask_cube = self.im_mask_cube[sel] * self.im_bg_mask[sel]
                else:
                    mask_cube = self.im_mask_cube[sel]
                psf_cube0, scale0, dbg0, w0 = multi_psf_fit(
                                self.eigen_psf[:params.klip],
                                self.im_sub[sel] - bg_mod[sel],
                                self.im_noise[sel],
                                mask_cube,
                                self.im_xc[sel], self.im_yc[sel],
                                fitrad=params.fitrad,
                                defrad=self.pps.psf_rad,
                                bg_fit=bg_fit,
                                nthreads=self.nthreads_reduced, 
                                non_negative=self.pps.non_neg_lsq)
                # Interpolate over frames without source
                t0 = self.im_att[sel, 0]
                t = self.im_att[:, 0]
                self.im_psf_cube = interp_cube_ext(t, t0, psf_cube0*self.im_apt)
                w = interp_cube_ext(t, t0, w0)
                scale = np.interp(t, t0, scale0)
                dbg = np.interp(t, t0, dbg0)
                self.im_dbg = dbg

                self.mess('Iter {:d} MAD im: {:.2f} ppm'.format(n+1, mad(scale0)))
                self.make_mask_cube_im()
                self.im_mask_cube[sel==0] = self.im_mask
                res = self.compute_residuals_im()

                if self.pps.smear_resid_im:
                    self.update_smear_im(res)
                    res = self.compute_residuals_im()
                if params.bStat:
                    self.compute_resid_stat_im(res)

                self.im_noise = (self.psf_noise_im(np.abs(self.im_psf_cube) +
                                                np.abs(self.bg_model_im()) +
                                                np.abs(self.im_dark))**2 +
                                self.im_dark_err**2)**.5
                
                if self.pps.bgstars and self.pps.fit_bgstars and self.im_bg_refined is False:
                    # Only refince once
                    self.refine_star_bg_im()
                    self.make_star_bg_cube_im()
                    self.im_bg_refined = True

            flux, err = self.psf_phot_im(self.im_psf_cube, self.pps.fitrad)
            flag = self.flag_im(scale, sel)

            sel = (flag==0)
            self.im_mad = mad(scale[sel])
            self.mess('MAD im: {:.2f} ppm'.format(mad(scale)))
            curr_mad = mad(scale[sel])
            self.mess('MAD im[flag==0]: {:.2f} ppm'.format(curr_mad))
            self.im_flux = scale
            self.im_sel = sel

            if curr_mad < self.im_best_mad:
                self.im_best_mad = curr_mad
                self.save_results_im(scale, err, w, flag)

            return scale, dbg, flux, err, sel, w

        klip = min(self.pps.klip, max_klip)
        nominal = TestParams(klip=klip, fitrad=self.pps.fitrad, bBG=(self.pps.bg_fit==0),
                                bDark=self.pps.darksub, bStat=self.pps.remove_static)

        if self.pps.im_optimise is False:
            scale, dbg, flux, err, sel, w = test_iter(nominal)
        else:
            self.mess('Start optimising {:s}/{:s} ({:05d}) [im]'.format(
                self.pps.name, self.pps.visit, self.pps.version), 0)            
            fo = FindOptimal(self.pps.im_test_klips, self.pps.im_test_fitrads,
                             self.pps.im_test_BG, self.pps.im_test_Dark, self.pps.im_test_Stat)
            for n in range(self.pps.optimise_restarts):
                nominal.mad = None
                current_mad = None
                fo.start(nominal, self.pps.optimise_tree_iter)
                while True:
                    testparams = fo.next_test(current_mad)
                    if testparams is None:
                        break
                    if self.pps.im_test_Dark:
                        self.pps.darksub = testparams.bDark
                        self.reduce_data_im()
                    scale, dbg, flux, err, sel, w = test_iter(testparams)
                    current_mad = self.im_mad
                    self.mess('{:s}, mad={:.2f} [im]'.format(str(testparams), current_mad), 0)
                nominal = fo.get_best()
                self.mess('--- Iter best {:d}: {:s} [im]'.format(n, str(nominal)), 0)
            self.mess('Optimisation done ({:d} iterations) [im]'.format(len(fo.tested_params)), 0)
            self.mess(fo.str_tested(), 0)
            self.mess('--- Generating best result [im]', 0)
            nominal.mad = None
            scale, dbg, flux, err, sel, w = test_iter(nominal, self.pps.sigma_clip_niter+1)
            self.mess('{:s}, mad={:.2f} [im]'.format(str(nominal), self.im_mad), 0)
            self.mess('--- Done! [im]', 0)

        return  scale, dbg, flux, err, sel, w


    def flag_sa(self, scale, sel):
        """Computes and return status flags for extracted
        photometry series (of subarrays).
        """
        flagCenter = (self.filter_pos(self.sa_xc, self.sa_yc) == 0)
        flagBadPix = (self.filter_bad_masks(self.sa_mask_cube, self.sa_apt) == 0)
        flagFlux = (self.filter_flux(scale) == 0)
        flagSource = (sel == 0)
        flagBG = (self.filter_high_bg(self.sa_dbg + self.sa_bg, sel=(flagFlux==0)) == 0)
        flag = np.zeros(flagCenter.shape, dtype='int8')
        flag[:] = (1*flagCenter + 2*flagBadPix + 4*flagFlux +
                    8*flagSource + 16*flagBG)
        return flag


    def flag_im(self, scale, sel):
        """Computes and return status flags for extracted
        photometry series (of imagettes).
        """
        flagCenter = (self.filter_pos(self.im_xc, self.im_yc) == 0)
        flagBadPix = (self.filter_bad_masks(self.im_mask_cube, self.im_apt) == 0)
        flagFlux = (self.filter_flux(scale) == 0)
        flagSource = (sel == 0)
        flagBG = (self.filter_high_bg(self.im_dbg + self.im_bg, sel=(flagFlux==0)) == 0)
        flag = np.zeros(flagCenter.shape, dtype='int8')
        flag[:] = (1*flagCenter + 2*flagBadPix + 4*flagFlux +
                    8*flagSource + 16*flagBG)
        return flag


    def save_results_sa(self, scale, err, w, flag):
        """Save data from reduction/extraction processes according to 
        switches of parameter file. 
        """        
        if self.pps.save_resid_cube:
            self.save_residuals_sa('')

        if self.pps.save_bg_cube:
            self.save_bg_sa('')

        if self.pps.save_bg_models:
            self.save_bg_model_sa('')

        self.save_eigen_sa(flag, scale, scale*err,
                           self.sa_dbg + self.sa_bg, w)

        if self.pps.save_mask_cube:
            self.save_cube_fits('mask_cube_sa.fits', 
                                np.array(self.sa_mask_cube, dtype='uint8'))

        if self.pps.save_psfmodel:
            self.save_cube_fits('psf_model_sa.fits', self.sa_psf_cube)

        if self.pps.save_static:
            self.save_cube_fits('static_sa.fits', self.sa_stat_res)

        if self.pps.save_noise_cubes:
            self.save_cube_fits('psf_noise_sa.fits', self.sa_noise)
            self.save_cube_fits('raw_noise_sa.fits', self.raw_noise_sa())
            res = self.compute_residuals_sa()
            emp_noise_cube = empiric_noise(res, self.sa_xc, self.sa_yc, self.sa_dbg + self.sa_bg)
            self.save_cube_fits('empiric_noise_sa.fits', emp_noise_cube)


    def save_results_im(self, scale, err, w, flag):
        """Save data from reduction/extraction processes according to 
        switches of parameter file. Return status flags for extracted
        photometry series.
        """        

        if self.pps.save_resid_cube:
            self.save_residuals_im('')

        if self.pps.save_bg_cube:
            self.save_bg_im('')

        if self.pps.save_bg_models:
            self.save_bg_model_sa('')

        self.save_eigen_im(flag, scale, scale*err,
                           self.im_dbg + self.im_bg, w)
        
        if self.pps.save_mask_cube:
            self.save_cube_fits('mask_cube_im.fits',
                                np.array(self.im_mask_cube, dtype='uint8'))

        if self.pps.save_psfmodel:
            self.save_cube_fits('psf_model_im.fits', self.im_psf_cube)

        if self.pps.save_static:
            self.save_cube_fits('static_im.fits', self.im_stat_res)

        if self.pps.save_noise_cubes:
            self.save_cube_fits('psf_noise_im.fits', self.im_noise)
            self.save_cube_fits('raw_noise_im.fits', self.raw_noise_im())
            res = self.compute_residuals_im()
            emp_noise_cube = empiric_noise(res, self.im_xc, self.im_yc, self.im_dbg + self.im_bg)
            self.save_cube_fits('empiric_noise_im.fits', emp_noise_cube)

        return flag



    def estimate_level_sa(self, debiased, remove_stars=False, level=0.01, outer_median=True):
        """Estimate background level self.sa_bg. If remove_stars, then remove
        the previously determined field stars. If outer_median, then adjust the
        level such median outer 90% of the aperture over all frames is zero. This
        is a way to reduce the influence of the PSF wings from the target.
        """
        if remove_stars:
            datacube = debiased - self.sa_norm*self.sa_bgstars - self.sa_psf_cube
            datacube[self.sa_bgstars > level] = np.nan
            datacube[self.sa_psf_cube/self.sa_norm > level] = np.nan
        else:
            datacube = debiased.copy()
        if self.pps.mask_badpix:
            datacube *= self.sa_mask_cube
        bg = level_estimate(datacube, nthreads=self.pps.nthreads)
        if outer_median: # Correct for target PSF wing contamination of bg estimate
            sel = sigma_clip(bg, clip=3, niter=10)
            bg += level_ring_median(datacube) - np.nanmedian(bg[sel])
        return bg


    def reduce_data_sa(self):
        """Background subtracts the de-biased frames. The central 50% of each 
        framed is checked for a source, to filter out source-less frames.
        If center is True, a photocentre is computed for each frame with 
        a source.
        """
        self.mess('--- Starting reductions [sa]')
        sa_sub = self.sa_debias.copy()

        if self.pps.darksub:
            self.mess('Subtracting dark [sa]')
            sa_sub -= self.sa_dark
            if self.pps.mask_bad_dark:
                bp = np.median(sa_sub, axis=0) < 0
                self.mess('Bad dark pixels: {:d} [sa]'.format(np.sum(bp)))
                self.sa_mask[bp] = 0
                self.sa_mask_cube[:] = self.sa_mask
        else:
            self.mess('Not subtracting dark [sa]', 2)

        self.mess('Estimating background levels [sa]')
        self.sa_bg = self.estimate_level_sa(sa_sub)
        self.mess('  Median: {:.2f} Std: {:.2f} Min: {:.2f} Max: {:.2f}'.format(
            np.nanmedian(self.sa_bg), np.nanstd(self.sa_bg),
            np.nanmin(self.sa_bg), np.nanmax(self.sa_bg)))
        sa_sub -= self.sa_apt*self.sa_bg[:, None, None]
        self.sa_sub = sa_sub
        
        self.sa_noise = self.raw_noise_sa()
        sel = self.has_source(sa_sub, mask=self.sa_mask)
        self.sa_cent_sel = sel
        self.mess('No centering source: {:d} / {:d} [sa]'.format(np.sum(sel==0), len(sel)))


    def reduce_data_im(self):
        """Reduce imagettes by subtracting bias, applying a non-linear
        correction, dividing by a flat field and subtracting a preliminary 
        background interpolated from the subarray background estimate.
        Subtract the dark current 
        Also compute a rough noise estimate and check if any frames are
        missing a source.
        If center is True, a photocentre is computed for each frame with 
        a source.
        """
        self.mess('--- Starting reductions [im]')
        im_sub = self.im_debias.copy()

        if self.pps.darksub:
            self.mess('Subtracting dark [im]')
            im_sub -= self.im_dark
            if self.pps.mask_bad_dark:
                bp = np.median(im_sub, axis=0) < 0
                self.mess('Bad dark pixels: {:d} [im]'.format(np.sum(bp)))
                self.im_mask[bp] = 0
                self.im_mask_cube[:] = self.im_mask
        else:
            self.mess('Not subtracting dark [im]', 2)

        self.mess('Inteprolating background levels from subarrays [im]')
        self.sa_bg2im_bg()

        im_sub -= self.im_apt*self.im_bg[:, None, None]
        self.im_sub = im_sub

        self.im_noise = self.raw_noise_im()
        sel = self.has_source(im_sub, mask=self.im_mask)
        self.im_cent_sel = sel
        self.mess('No centering source: {:d} / {:d} [im]'.format(np.sum(sel==0), len(sel)))


    def init_target_coo(self, shape):
        """Initialises target pixel coordinates to center of array,
        plus offset defined by pps.centre_off_x, pps.centre_off_y
        Returns 1D xc and yc arrays.
        """
        xc = 0.5*shape[2]*np.ones(shape[0]) + self.pps.centre_off_x
        yc = 0.5*shape[1]*np.ones(shape[0]) + self.pps.centre_off_y
        return xc, yc


    def read_subarrays(self):
        """Reads the raw subarray data cube from data file, removes bias, 
        divides flat field, corrects non-linearity and CTI. Also initialises
        variables that depend on the properties of the loaded data.
        """
        self.mess('Reading raw data cube [sa]')
        
        sa_raw, self.sa_mjd, self.sa_hdr, self.sa_tab = \
            read_raw_datacube(self.pps.file_sa_raw, self.pps.sa_range)
        self.sa_nexp = self.sa_hdr['NEXP']
        self.sa_off = (self.sa_hdr['X_WINOFF'], self.sa_hdr['Y_WINOFF'])

        # Define aperture mask
        self.sa_apt = np.isfinite(sa_raw[0])

        # Define centre coordinates of frames in data cube
        self.sa_xc, self.sa_yc = self.init_target_coo(sa_raw.shape)

        # Initialise data cube with star background
        self.sa_bgstars = np.zeros_like(sa_raw)

        # Initialise data cube for smear (# frames x width)
        self.sa_smear = np.zeros(sa_raw.shape[0:3:2])
        self.sa_smear_resid = np.zeros(sa_raw.shape[0:3:2])

        # Initialise frame for static residual image
        self.sa_stat_res = 0 * sa_raw[0]

        # Mutliply with gain
        gain = self.gain_fun(self.sa_mjd) * np.ones_like(self.sa_mjd)
        sa_raw *= gain[:,None,None]
        
        # Subtract bias
        sa_raw -= self.bias*self.sa_nexp

        # Flatfield
        if self.pps.flatfield:
            self.mess('Applying flatfield (Teff={:.1f}K) [sa]'.format(self.pps.Teff))
            flat = self.get_flatfield(self.pps.Teff, self.sa_off, sa_raw[0].shape)
            sa_raw /= flat
        else:
            self.mess('Not applying flatfield correction. [sa]')

        # CTI correction
        if self.pps.cti_corr:
            self.update_cti_fun(np.mean(self.sa_mjd))        
            self.mess('Applying CTI correction. [sa]')
            sa_raw *= self.cti_corr_fun(sa_raw / gain[:,None,None] / self.sa_nexp)
        else:
            self.mess('Not applying CTI correction. [sa]')

        # Non-linearity correction
        if self.pps.non_lin:
            self.nonlinfun = nonlinear(self.pps.file_nonlin)
            self.mess('Correcting non-linearity [sa]')
            sa_raw *= self.nonlinfun(sa_raw / gain[:,None,None] / self.sa_nexp)
        else:
            self.mess('No correction for non-linearity. [sa]')
        self.sa_debias = sa_raw



    def read_imagettes(self):
        """Reads the raw imagette data cube from data file, removes bias, 
        divides flat field, corrects non-linearity and CTI. Also initialises
        variables that depend on the properties of the loaded data.
        """
        self.mess('Reading imagette data cube')

        # Define range of imagettes to read, if relevant
        if self.pps.sa_range is not None:
            _raw, _mjd, hdr, _tab = read_raw_datacube(self.pps.file_im, (0,1))
            nexp = self.sa_nexp / hdr['NEXP']
            im_range = (int(self.pps.sa_range[0]*nexp), 
                        int(self.pps.sa_range[1]*nexp))
        else:
            im_range = None 

        im_raw, self.im_mjd, self.im_hdr, self.im_tab = \
            read_raw_datacube(self.pps.file_im, im_range)

        # Define nexp and offsets
        self.im_nexp = self.im_hdr['NEXP']
        self.nexp = self.sa_nexp / self.im_nexp
        self.im_off, self.im_sa_off = imagette_offset(self.pps.file_im)
        
        # Define aperture mask
        self.im_apt = np.isfinite(im_raw[0])

        # Define centre coordinates of frames in data cube
        self.im_xc, self.im_yc = self.init_target_coo(im_raw.shape)

        # Initialise data cube with star background
        self.im_bgstars = np.zeros_like(im_raw)

        # Initialise data cube for smear (# frames x width)
        self.im_smear = np.zeros(im_raw.shape[0:3:2])
        self.im_smear_resid = np.zeros(im_raw.shape[0:3:2])

        # Initialise frame for static residual image
        self.im_stat_res = 0 * im_raw[0]

        # Mutliply with gain
        gain = self.gain_fun(self.im_mjd) * np.ones_like(self.im_mjd)
        im_raw *= gain[:,None,None]
        
        # Subtract bias
        im_raw -= self.bias*self.im_nexp

        # Flatfield
        if self.pps.flatfield:
            self.mess('Applying flatfield (Teff={:.1f}K) [im]'.format(self.pps.Teff))
            flat = self.get_flatfield(self.pps.Teff, self.im_off, im_raw[0].shape)
            im_raw /= flat
        else:
            self.mess('Not applying flatfield correction. [im]')

        # CTI correction
        if self.pps.cti_corr:
            self.update_cti_fun(np.mean(self.im_mjd))        
            self.mess('Applying CTI correction. [im]')
            im_raw *= self.cti_corr_fun(im_raw / gain[:,None,None] / self.im_nexp)
        else:
            self.mess('Not applying CTI correction. [im]')

        # Non-linearity correction
        if self.pps.non_lin:
            self.nonlinfun = nonlinear(self.pps.file_nonlin)
            self.mess('Correcting non-linearity [im]')
            im_raw *= self.nonlinfun(im_raw / gain[:,None,None] / self.im_nexp)
        else:
            self.mess('No correction for non-linearity. [im]')
        self.im_debias = im_raw



    def read_mask(self):
        """Reads and interprets reference files of bad pixel maps. The bad pixel
        maps are defined such that -2 are deade pixels, -1 "half-dead", 0 are 
        good, 1 are hot, 2 are saturated, and 3 are telegraphic. For the mask levels,
         0 masks nothing,
        -2 masks only dead,
        -1 masks dead and half dead,
        3 masks dead, half dead and telegraphic,
        2 masks dead, half dead, telegraphic, and saturated
        1 masks dead, half dead, telegraphic, saturated, and hot pixels.
        The mask has values 0 for masked and 1 for non-masked pixels.
        The mask is propagated to the imagettes, if relevant.
        """
        self.sa_mask = self.sa_apt.copy()
        self.sa_mask_cube = np.ones(self.sa_debias.shape, dtype='?')
        if self.pps.mask_badpix:
            self.mess(f'Load mask [{self.pps.mask_level}]')
            mask, filename = read_bad(self.pps.calibpath, np.median(self.sa_mjd), self.sa_off, self.sa_mask.shape)
            self.mess('Bad pixelmap \'{:s}\''.format(filename))
            if self.pps.mask_level in [-2, -1, 3, 2, 1]:
                self.sa_mask *= (mask!=-2)
            if self.pps.mask_level in [-1, 3, 2, 1]:
                self.sa_mask *= (mask!=-1)
            if self.pps.mask_level in [3, 2, 1]:
                self.sa_mask *= (mask!=3)
            if self.pps.mask_level in [2, 1]:
                self.sa_mask *= (mask!=2)
            if self.pps.mask_level in [1]:
                self.sa_mask *= (mask!=1)
            self.mess('{:d} pixels masked in subarray'.format(np.sum(self.sa_mask[self.sa_apt]==0)))
            self.sa_mask_cube[:] = self.sa_mask

            if self.pps.file_im is not None:
                self.define_im_mask()
        else:
            self.mess('Mask not defined', level=2)


    def define_im_mask(self):
        """Crops the imagette part of the subarray mask to define the
        imagette mask.
        """
        self.im_mask = self.sa2im_crop(self.sa_mask) * self.im_apt
        self.im_mask_cube = np.ones(self.im_debias.shape, dtype='?')
        self.im_mask_cube[:] = self.im_mask
        self.mess('{:d} pixels masked in imagettes'.format(np.sum(self.im_mask[self.im_apt]==0)))


    def read_darks(self):
        """Read dark current estimates from directory with dark current files
        If no dark current is to be subtracted, define it to be zero.
        """
        if not self.pps.darksub:
            self.mess('Skipping dark frames')
            self.sa_dark = np.zeros_like(self.sa_debias[0])
            self.sa_dark_err = self.sa_dark
            if self.pps.file_im is not None:
                self.im_dark = np.zeros_like(self.im_debias[0])
                self.im_dark_err = self.im_dark
            return
        self.mess('Reading dark frames')
        dark, dark_err, filename0, filename1 = read_dark(self.pps.calibpath, np.median(self.sa_mjd),
                                                         self.sa_off, self.sa_apt.shape)
        self.mess('First dark frame \'{:s}\''.format(filename0))
        self.mess('Second dark frame \'{:s}\''.format(filename1))

        # Remove pixels for dark current correction
        sel = (dark/dark_err < self.pps.dark_min_snr) * (dark < self.pps.dark_min_level)
        dark[sel] = 0
        dark_err[sel] = 0

        if self.pps.cti_corr:
            self.mess('Applying CTI correction to dark.')
            dark *= self.cti_corr_fun(dark * self.sa_hdr['EXPTIME'] / self.gain_fun())

        self.sa_dark = dark * self.sa_hdr['TEXPTIME'] * self.sa_apt
        self.sa_dark_err = dark_err * self.sa_hdr['TEXPTIME'] * self.sa_apt
        self.mess('Dark current defined for {:d}/{:d} = {:.2f}% pixels [sa]'.format(
            np.sum(self.sa_dark > 0), np.sum(self.sa_apt > 0),
            100.0 * np.sum(self.sa_dark > 0) / np.sum(self.sa_apt > 0)))

        if self.pps.file_im is not None:
            i0, i1, j0, j1 = sub_image_indices(self.im_sa_off, self.im_apt.shape)
            self.im_dark = (dark[j0:j1, i0:i1] * self.im_nexp *
                            self.im_hdr['EXPTIME'] * self.im_apt)
            self.im_dark_err = (dark_err[j0:j1, i0:i1] * self.im_nexp *
                                self.im_hdr['EXPTIME'] * self.im_apt)
            self.mess('Dark current defined for {:d}/{:d} = {:.2f}% pixels [im]'.format(
                np.sum(self.im_dark > 0), np.sum(self.im_apt > 0),
                100.0 * np.sum(self.im_dark > 0) / np.sum(self.im_apt > 0)))


    def read_attitude(self):
        """Reads attitude data from attitude file and resamples it
        to imagette cadence if imagettes are defined
        """
        self.mess('Reading attitude data')
        att = attitude(self.pps.file_att)
        self.sa_t = (att[:,0]-att[0,0])/(att[-1,0]-att[0,0])
        oexptime = max(self.sa_hdr['TEXPTIME'], self.sa_hdr['EXPTIME'])
        self.sa_att = resample_attitude(self.sa_mjd, oexptime, att)
        self.sa_t = ((self.sa_att[:,0]-self.sa_att[0,0]) /
                     (self.sa_att[-1,0]-self.sa_att[0,0]))
        if self.pps.file_im is not None:
            iexptime = max(self.im_hdr['TEXPTIME'], self.im_hdr['EXPTIME'])
            self.im_att = resample_attitude(self.im_mjd, iexptime, att)
            self.im_t = ((self.im_att[:,0]-self.im_att[0,0]) /
                         (self.im_att[-1,0]-self.im_att[0,0]))


    def make_gain_function(self):
        """Defines the gain in e/ADU for each imagette to either a
        given value or computed from HK. Returns gain function
        g(mjd) in electrons/ADU.
        """
        if self.pps.gain is None:
            self.mess('Reading HK and computing gain')
            mjd, gain = read_gain(self.pps.file_hk, self.pps.file_gain)
            mjdC = np.nanmedian(mjd)
            self.mess('Estimated gain = {:.4f} e/ADU +/- {:.1f} ppm'.format(np.nanmedian(gain),
                            1e6*np.nanstd(gain)/np.nanmedian(gain)))
            if self.pps.save_gain:
                self.save_cube_fits('gain.fits', np.array([mjd, gain]).T)
            if self.pps.gain_median:
                self.gain_fun = lambda x=0, median_gain=np.nanmedian(gain): median_gain
            else:
                self.gain_fun = lambda x=mjdC, imjd=mjd,igain=gain: np.interp(x, imjd, igain)
        else:
            self.gain_fun = lambda x=0, def_gain=self.pps.gain: def_gain
            self.mess('Defined gain = {:.4f} e/ADU'.format(self.pps.gain))


    def read_bias_ron(self):
        """Finds bias and read-out noise from raw subarray file.
        Gain must first be defined.
        """
        self.mess('Estimating bias and readout noise')
        if self.gain_fun is None:
            raise Exception('No gain defined')
        gain = self.gain_fun()

        if self.pps.ron is None or self.pps.bias is None:
            self.mess('Reading subarray bias areas')
            self.bias, self.ron = bias_ron_adu(self.pps.file_sa_raw, gain)
        else:
            self.mess('Defining bias and read-out noise from input parameters')
            self.ron = self.pps.ron * gain
            self.bias = self.pps.bias * gain
        self.mess('RON = {:.2f} ADU = {:.2f} elec'.format(self.ron/gain, self.ron))
        self.mess('BIAS = {:.2f} ADU = {:.2f} elec'.format(self.bias/gain, self.bias))


    def read_thermFront_2(self):
        """Reads the thermFront_2 sensor data from the raw subarray
        data cube.
        """
        self.mess('Reading thermFront_2 sensor data')
        self.sa_thermFront_2 = thermFront_2(self.pps.file_sa_raw)
        if self.pps.sa_range is not None:
            self.sa_thermFront_2 = self.sa_thermFront_2[self.pps.sa_range[0]:self.pps.sa_range[1]]
        if self.pps.file_im is not None:
            self.im_thermFront_2 = np.interp(self.im_t, self.sa_t,
                                             self.sa_thermFront_2)


    def read_starcat(self):
        """Initiates synstar object from star catalogue file
        """
        if not self.pps.bgstars:
            self.mess('Skipping star catalogue')
            return
        self.mess('Loading star catalogue')
        b = self.sa_debias.shape[2]
        h = max(self.sa_hdr['Y_WINOFF'] + self.sa_debias.shape[1],
                self.pps.ccdsize[0] - self.sa_hdr['Y_WINOFF'])
        maxrad = (b**2 + h**2)**0.5
        self.starcat = star_bg(self.pps.file_starcat,
                                        psf_lib=self.psf_lib,
                                        maxrad=maxrad,
                                        fscalemin=self.pps.limflux,
                                        pixel_scale=self.pps.pixel_scale,
                                        star_rad_scale = self.pps.star_rad_scale)
        self.mess(f'{self.starcat.catsize} stars loaded')


    def define_workcat(self):
        """Define star catalagoues tuned to each frame. Used when producing
        background stars, including rotational blur.
        """
        if not self.pps.bgstars:
            self.mess('Skipping working catalogue definition')
            return
        self.mess('Defining working star catalogues [sa]')
        self.sa_workcat = []
        for n in range(len(self.sa_debias)):
            maxrad = 0.5*np.max(self.sa_debias[0].shape) + self.pps.sa_psfrad
            self.sa_workcat.append(self.starcat.image_cat(self.sa_xc[n],
                                                          self.sa_yc[n],
                                                          self.sa_att[n,3],
                                                          self.sa_att[n,4],
                                                          maxrad=int(maxrad),
                                                          resolution=self.pps.blur_res))
        if self.pps.file_im is not None:
            self.mess('Defining working star catalogues [im]')
            self.im_workcat = []
            for n in range(len(self.im_debias)):
                maxrad = 0.5*np.max(self.im_debias[0].shape) + self.pps.sa_psfrad
                self.im_workcat.append(self.starcat.image_cat(self.im_xc[n],
                                                            self.im_yc[n],
                                                            self.im_att[n,3],
                                                            self.im_att[n,4],
                                                            maxrad=int(maxrad),
                                                            resolution=self.pps.blur_res))


    def make_mjd2bjd_function(self):
        """ Define function that converts between modified Julian date
        and barycentric Julian date, assuminging known target coordinates. 
        """
        self.mess('Defining MJD to BJD conversion')
        if self.pps.mjd2bjd:
            self.mjd2bjd = (lambda mjd, ra=self.sa_hdr['RA_TARG'],
                            dec=self.sa_hdr['DEC_TARG'] : mjd2bjd(mjd, ra, dec))
        else: # Don't use barycentric conversion
            self.mjd2bjd = lambda mjd : mjd + 2400000.5

        
    def adu2e_im(self, im_adu):
        """Applies a non-linearity function and the gain to convert from
        detected ADU to number of electrons.
        """
        if self.pps.cti_corr:
            self.mess('Correcting for CTI [im]')
            im_adu *= self.cti_corr_fun(im_adu/self.im_hdr['NEXP'])
        self.mess('Converting ADU to electrons [im]')
        if self.pps.non_lin:
            self.mess(' Using non-linearity correction [im]')
            nonlinfun = nonlinear(self.pps.file_nonlin)
            im_e = im_adu * nonlinfun(im_adu/self.im_hdr['NEXP']) * self.im_gain[:,None,None]
        else:
            self.mess(' No non-linearity correction [im]', level=2)
            im_e = im_adu  * self.im_gain[:,None,None]            

        return im_e


    def get_flatfield(self, target_Teff, offset, shape):
        """Retrieves flatfield reference file to correct data for flatfield.
        target_Teff is the target radiation temperature, used to produce
        a weighted (wavelength-dependent) flatfield. offset is the offset in
        pixels of frame within full detector size, and shape is shape of
        flatefield. Returns flatfield
        """
        return flatfield(self.pps.file_flats, target_Teff, offset, shape)


    def smear_limit(self):
        """Estimates how bright stars need to be to sighnificantly
        smear (used to cull star sample and thus limit computing time)
        """
        return (self.sa_hdr['EXPTIME'] *
                self.sa_hdr['RO_FREQU'] * 
                self.pps.smear_const)
        

    def compute_smearing_sa(self):
        """Compute the smearing correction using stars from the
        star catalogue, and the given subarray PSF.
        """
        self.sa_smear = np.zeros(self.sa_debias.shape[0:3:2])
        if not self.pps.bgstars or not self.pps.smear_corr:
            self.mess('No smearing correction update [sa].', level=2)
            return
        limflux = self.smear_limit()
        num_bright = np.sum(self.starcat.fscale>=limflux)
        self.mess('Smear limit flux: {:.2e} ({:d} stars) [sa]'.format(limflux, num_bright))

        if num_bright == 0:
            return

        yoff = self.sa_hdr['Y_WINOFF']
        shape = (self.pps.ccdsize[0], self.sa_sub.shape[2])
        for n in range(len(self.sa_sub)):
            xc = self.sa_xc[n]
            yc = yoff + self.sa_yc[n]
            self.sa_smear[n] = self.pps.smear_fact * self.starcat.smear(xc, yc,
                                                          self.sa_att[n,3],
                                                          shape=shape,
                                                          limflux=limflux)
        self.sa_smear /= (self.sa_hdr['EXPTIME']*self.sa_hdr['RO_FREQU'])


    def compute_smearing_im(self):
        """Compute the smearing correction using stars from the
        star catalogue, and the given imagette PSF.
        """
        self.im_smear = np.zeros(self.im_debias.shape[0:3:2])
        if not self.pps.bgstars or not self.pps.smear_corr:
            self.mess('No smearing correction update [im].', level=2)
            return

        limflux = self.smear_limit()
        num_bright = np.sum(self.starcat.fscale>=limflux)
        self.mess('Smear limit flux: {:.2e} ({:d} stars) [im]'.format(limflux, num_bright))

        if num_bright == 0:
            return

        yoff = self.im_off[1]
        shape = (self.pps.ccdsize[0], self.im_sub.shape[2])
        for n in range(len(self.im_sub)):
            xc = self.im_xc[n]
            yc = yoff + self.im_yc[n]
            self.im_smear[n] = self.pps.smear_fact * self.starcat.smear(xc, yc,
                                                          self.im_att[n,3],
                                                          shape=shape,
                                                          limflux=limflux)
        self.im_smear /= (self.im_hdr['EXPTIME'] * self.sa_hdr['RO_FREQU'])


    def compute_resid_stat_sa(self, res):
        """Compute the static part of the residuals
        """
        nanres = res.copy()
        apts = self.sa_apt
        nanres[:,apts==0] = np.nan
        np.nan_to_num(nanres[0,:,:], copy=False)   # Ensures not all values are nan
        self.sa_stat_res += np.nanmedian(nanres, axis=0)
        if self.pps.pos_static:
            self.sa_stat_res *= (self.sa_stat_res > 0)


    def compute_resid_stat_im(self, res):
        """Compute the static part of the residuals
        """
        nanres = res.copy()
        apts = self.im_apt
        nanres[:, apts==0] = np.nan
        np.nan_to_num(nanres[0,:,:], copy=False)   # Ensures not all values are nan
        self.im_stat_res += np.nanmedian(nanres, axis=0)
        if self.pps.pos_static:
            self.im_stat_res *= (self.im_stat_res > 0)


    def update_smear_sa(self, res):
        """Looks at resduals along columns to remove any residual 
        smearing. Uses sigma-clipping to remove outliers.
        """
        self.mess('Removing residual smear [sa]')
        for n in range(len(res)):
            self.sa_smear_resid[n] += resid_smear(res[n])/self.sa_norm


    def update_smear_im(self, res):
        """Looks at resduals along columns to remove any residual 
        smearing. Uses sigma-clipping to remove outliers.
        Warning: this can result in a noisy correction for imagettes
        where the field outside PSF is small.
        """
        self.mess('Removing residual smear [im]')
        for n in range(len(res)):
            self.im_smear_resid[n] += resid_smear(res[n])/self.im_norm


    def sa_bg2im_bg(self):
        """Define an initial background estimate for the imagettes by
        interpolating the already known background for the subarrays.
        """
        self.mess('Interpolate background from subarray to imagettes')
        self.im_bg = np.interp(self.im_mjd, self.sa_mjd, self.sa_bg) / self.nexp


    def raw_noise_sa(self):
        """Compute noise from debiased data, assuming poisson noise, 
        read-out noise, and dark current noise (when applicable)
        """
        self.mess('Compute noise cube [sa]')
        iron = self.ron*self.sa_nexp**0.5
        raw_noise = noise(self.sa_debias, ron_elec=iron)
        if self.pps.darksub:
            raw_noise = (raw_noise**2 + self.sa_dark_err**2)**.5
        # Define noise outside aperture
        raw_noise = np.nan_to_num(raw_noise)
        raw_noise[raw_noise==0] = np.max(np.median(raw_noise, axis=0))
        return raw_noise


    def raw_noise_im(self):
        """Compute noise from debiased data, assuming poisson noise, 
        read-out noise, and dark current noise (when applicable)
        """
        self.mess('Compute noise cube [im]')
        iron = self.ron*self.im_nexp**0.5
        raw_noise = noise(self.im_debias, ron_elec=iron)
        if self.pps.darksub:
            raw_noise = (raw_noise**2 + self.im_dark_err**2)**.5
        # Define noise outside aperture
        raw_noise = np.nan_to_num(raw_noise)
        raw_noise[raw_noise==0] = np.max(np.median(raw_noise, axis=0))
        return raw_noise


    def psf_noise_sa(self, src_mod_cube):
        """Compute noise assuming a model distribution of flux
        """
        self.mess('Compute PSF noise [sa]')
        iron = self.ron*self.sa_nexp**0.5
        mod_noise = psf_noise(src_mod_cube, ron_elec=iron)
        # Define noise outside aperture
        mod_noise = np.nan_to_num(mod_noise)
        mod_noise[mod_noise==0] = np.max(np.median(mod_noise, axis=0))
        return mod_noise


    def psf_noise_im(self,  src_mod_cube):
        """Compute noise assuming a model distribution of flux
        """
        self.mess('Compute PSF noise [im]')
        iron = self.ron*self.im_nexp**0.5
        mod_noise = psf_noise(src_mod_cube, ron_elec=iron)
        # Define noise outside aperture
        mod_noise = np.nan_to_num(mod_noise)
        mod_noise[mod_noise==0] = np.max(np.median(mod_noise, axis=0))
        return mod_noise

        
    def photo_cent(self, data, radius=None):
        """Find photo centre for each frame of a cube, using an aperture
        centered in the frame (with defined radius)
        """
        if radius == None:
            radius = self.pps.fitrad
        self.mess('Compute photo centers')
        apt = aperture(data[0].shape, radius=radius)
        ixc, iyc = cent_flux(data*apt)
        self.mess(' Cent std: ({:.3f}, {:.3f})'.format(np.std(ixc),
                                                  np.std(iyc)))
        return ixc, iyc


    def cent_deconv(self, datacube):
        """Find centre in pixel coordinates for target within pre-defined
        radius (pps.centfit). Uses centre of flux on brightest point source
        in area in deconvolved image. Returns x, y pixel coordinates of centre.
        """
        self.mess('Compute centroids  (multi {:d} threads)'.format(self.pps.nthreads))
        xi = int(datacube[0].shape[1]*0.5)
        yi = int(datacube[0].shape[0]*0.5)
        xc, yc = multi_cent_deconvolve(self.centre_psf,
                                  datacube,
                                  xi, yi,
                                  radius=self.pps.centfit_rad,
                                  subrad=self.pps.centfit_subrad,
                                  nthreads=self.pps.nthreads)
        ind = sigma_clip(xc) * sigma_clip(yc)
        self.mess('SA cent std: ({:.3f}, {:.3f})'.format(np.std(xc[ind]),
                                                      np.std(yc[ind])))
        return xc, yc


    def psf_cent_sa(self):
        """Use PSF fitting to find centres in subarrays. Interpolate positions
        over frames without source.
        """
        sel = self.sa_cent_sel        
        self.mess('Compute PSF centers [sa] (multi {:d} threads)'.format(self.pps.nthreads))

        # Experimental
        xi = int(self.sa_sub[0].shape[1]*0.5)
        yi = int(self.sa_sub[0].shape[0]*0.5)
        xc, yc = multi_cent_deconvolve(self.centre_psf,
                                  self.sa_sub[sel] - self.sa_stat_res,
                                  xi, yi,
                                  radius=15,
                                  subrad=3,
                                  nthreads=self.pps.nthreads)
#        xc, yc = multi_cent_psf(self.psf,
#                                self.sa_sub[sel] - self.sa_stat_res,
#                                self.sa_noise[sel],
#                                self.sa_xc[sel], self.sa_yc[sel],
#                                mask=self.sa_mask, radius=self.pps.fitrad,
#                                norm=self.sa_norm,
#                                nthreads=self.pps.nthreads)
        self.sa_xc = np.interp(self.sa_att[:, 0], self.sa_att[sel, 0], xc)
        self.sa_yc = np.interp(self.sa_att[:, 0], self.sa_att[sel, 0], yc)

        ind = sigma_clip(xc) * sigma_clip(yc)
        self.mess('SA cent std: ({:.3f}, {:.3f})'.format(np.std(xc[ind]),
                                                     np.std(yc[ind])))
        

    def psf_cent_im(self):
        """Use PSF fitting to find centres in imagettes. Interpolate positions
        over frames without source.
        """
        sel = self.im_cent_sel
        self.mess('Compute PSF centers [im] (multi {:d} threads)'.format(self.pps.nthreads))

        # Experimental
        xi = int(self.im_sub[0].shape[1]*0.5)
        yi = int(self.im_sub[0].shape[0]*0.5)
        xc, yc = multi_cent_deconvolve(self.centre_psf,
                                  self.im_sub[sel] - self.im_stat_res,
                                  xi, yi,
                                  radius=15,
                                  subrad=3,
                                  nthreads=self.pps.nthreads)

#        xc, yc = multi_cent_psf(self.psf,
#                                self.im_sub[sel] - self.im_stat_res, 
#                                self.im_noise[sel],
#                                self.im_xc[sel], self.im_yc[sel],
#                                mask=self.im_mask, radius=self.pps.fitrad,
#                                norm=self.im_norm,
#                                nthreads=self.pps.nthreads)
        self.im_xc = np.interp(self.im_att[:,0], self.im_att[sel, 0], xc)
        self.im_yc = np.interp(self.im_att[:,0], self.im_att[sel, 0], yc)
        ind = sigma_clip(xc) * sigma_clip(yc)
        self.mess('IM cent std: ({:.3f}, {:.3f})'.format(np.std(xc[ind]),
                                                     np.std(yc[ind])))       


    def make_pixtab_sa(self, sel, mask, fscale=None):
        """Produce a table of coordinates relative PSF centre, flux, and flux
        error per pixel. Used to derive a PSF.
        """
        self.mess('Produce pixel table [sa]')
        bg_model = self.bg_model_sa()
        return pix_mat(self.sa_sub[sel] - bg_model[sel], self.sa_noise[sel],
                              self.sa_xc[sel], self.sa_yc[sel], mask, fscale)

    
    def make_pixtab_im(self, sel, mask, fscale=None):
        """Produce a table of coordinates relative PSF centre, flux, and flux
        error per pixel. fscale is a normalisation array for the flux and the
        error, to put them on equal scale. Used to derive a PSF.
        """
        self.mess('Produce pixel table [im]')
        bg_model = self.bg_model_im()
        return pix_mat(self.im_sub[sel] - bg_model[sel], self.im_noise[sel],
                              self.im_xc[sel], self.im_yc[sel], mask, fscale)



    def median_flux(self, datacube, radius=25):
        """Computes the median flux of the source within the
        aperture radius. Used to scale PSF of expected
        background sources.
        """
        flux = self.apt_phot(datacube, radius=radius)
        sel = self.filter_flux(flux)
        return np.median(flux[sel])


    def apt_phot(self, datacube, radius=25):
        """Computes rough aperture photometry (i.e. fixed aperture,
        simple interpolation of bad pixels) of radius on each subarray. 
        Returns array of flux.
        """
        apt = aperture(datacube[0].shape, radius=radius)
        flux = np.nansum(datacube*apt, axis=(1,2))
        return flux

        
    def psf_phot_sa(self, psf_cube, radius):
        """Uses pre-defined PSFs to compute a noise-weighted sum of the flux
        of each subarray. Not as accurate as PSF fitting, but provides a
        good estimate of the expected error. Returns arrays of flux and
        its error estimate.
        """
        apt = self.apt_cube_sa(radius) * self.sa_mask_cube
        sa_noise = np.maximum(self.raw_noise_sa(), self.sa_noise)
        f, e = psf_phot_cube(self.sa_sub-self.bg_model_sa(), sa_noise, psf_cube, apt)
        return f, e


    def psf_phot_im(self, psf_cube, radius):
        """Uses pre-defined PSFs to compute a noise-weighted sum of the flux
        of each imagette. Not as accurate as PSF fitting, but provides a
        good estimate of the expected error. Returns arrays of flux and
        its error estimate.
        """
        apt = self.apt_cube_im(radius) * self.im_mask_cube
        im_noise = np.maximum(self.raw_noise_im(), self.im_noise)
        f, e = psf_phot_cube(self.im_sub-self.bg_model_im(), im_noise, psf_cube, apt)
        return f, e
        

    def update_cti_fun(self, epoch_mjd):
        """Initialises the CTI correction function at the given epoch (in MJD)
        according to the defined PIPE parameters (pps)
        """
        self.cti_corr_fun = cti_corr_fun(t=epoch_mjd,
                                         cti_t0=self.pps.cti_t0,
                                         cti_scale=self.pps.cti_scale,
                                         cti_expo=self.pps.cti_expo,
                                         cti_lim=self.pps.cti_lim)


    def apt_cube_sa(self, radius):
        """Returns a cube of binary apertures of radius centred according to 
        previously determined PSF offsets in each frame.
        """
        apt = np.zeros(self.sa_sub.shape, dtype='?')
        shape = apt[0].shape
        for n in range(len(apt)):
            apt[n] = aperture(shape, radius, self.sa_xc[n], self.sa_yc[n])
        return apt

        
    def apt_cube_im(self, radius):
        """Returns a cube of binary apertures of radius centred according to 
        previously determined PSF offsets in each frame.
        """
        apt = np.zeros(self.im_debias.shape, dtype='?')
        shape = apt[0].shape
        for n in range(len(apt)):
            apt[n] = aperture(shape, radius, self.im_xc[n], self.im_yc[n])
        return apt


    def make_mask_cube_sa(self):
        """Use a model of how the data should look like, the expected
        noise, and sigma-clipping to mask too deviating pixels 
        (e.g. cosmic rays). Only look for bad pixels inside radius.
        """
        if not self.pps.mask_badpix:
            self.mess('No mask cube [sa].', level=2)
            return
        self.mess('Make mask cube [sa]')
        if self.pps.empiric_noise:
            clip = self.pps.empiric_sigma_clip
            self.mess('Using empiric noise for mask cube (clip={:.1f}) [sa]'.format(clip))
            res = self.compute_residuals_sa()
            noise_cube = empiric_noise(res, self.sa_xc, self.sa_yc, self.sa_dbg + self.sa_bg)
        else:
            noise_cube = self.sa_noise
            clip = self.pps.sigma_clip

        self.sa_mask_cube = make_maskcube(self.sa_sub, noise_cube,
                                          self.sa_psf_cube + self.bg_model_sa(),
                                          mask=self.sa_mask, clip=clip)


    def make_mask_cube_im(self):
        """Use a model of how the data should look like, the expected
        noise, and sigma-clipping to mask too deviating pixels 
        (e.g. cosmic rays). Only look for bad pixels inside radius.
        """
        if not self.pps.mask_badpix:
            self.mess('No mask cube [im].', level=2)
            return
        self.mess('Make mask cube [im]')
        if self.pps.empiric_noise:
            clip = self.pps.empiric_sigma_clip
            self.mess('Using empiric noise for mask cube (clip={:.1f}) [im]'.format(clip))
            res = self.compute_residuals_im()
            noise_cube = empiric_noise(res, self.im_xc, self.im_yc, self.im_dbg + self.im_bg)
        else:
            noise_cube = self.im_noise
            clip = self.pps.sigma_clip
        self.im_mask_cube = make_maskcube(self.im_sub, noise_cube,
                                          self.im_psf_cube + self.bg_model_im(),
                                          mask=self.im_mask, clip=clip)


    def make_star_bg_cube_sa(self, skip=[0]):
        """Produces image cube of properly located background stars, to
        be subtracted from observations. skip is a list of stars to be skipped;
        typically the target itself (0), but also the secondary in case of
        a binary.
        """
        if not self.pps.bgstars:
            self.mess('No background stars computed [sa].', level=2)
            return
        self.mess('Model star background cube [sa]')
        self.sa_bgstars = make_star_bg(self.sa_sub[0].shape, self.starcat.psf_ids,
                                       self.starcat.psfs, self.sa_workcat,
                                       skip,
                                       krn_scl=self.pps.motion_step,
                                       krn_rad=self.pps.motion_nsteps,
                                       nthreads=self.pps.nthreads) * self.sa_apt



    def make_star_bg_cube_im(self, skip=[0]):
        """Produces image cube of properly located background stars, to
        be subtracted from observations. skip is a list of stars to be skipped;
        typically the target itself (0), but also the secondary in case of
        a binary.
        """
        if not self.pps.bgstars:
            self.mess('No background stars computed [im].', level=2)
            return
        self.mess('Model star background cube [im]')
        self.im_bgstars = make_star_bg(self.im_sub[0].shape, self.starcat.psf_ids,
                                       self.starcat.psfs, self.im_workcat,
                                       skip,
                                       krn_scl=self.pps.motion_step,
                                       krn_rad=self.pps.motion_nsteps,
                                       nthreads=self.pps.nthreads) * self.im_apt
        

    def make_bg_mask_sa(self, skip=[0]):
        """Produces mask over stars image cube of properly located background stars, to
        be subtracted from observations. skip is a list of stars to be skipped;
        typically the target itself (0), but also the secondary in case of
        a binary. Also excludes stars closer than pps.mask_bg_star_sep from target.
        """
        if not self.pps.bgstars or not self.pps.mask_bg_stars:
            self.mess('No background stars masked [sa]')
            return

        for n in range(1, self.starcat.catsize):
            if self.starcat.xpos[n]**2 + self.starcat.ypos[n]**2 <= self.pps.mask_bg_star_sep**2:
                skip.append(n)
            else:
                break
        skip = np.unique(skip)

        self.mess(f'Mask background stars; skip {skip} [sa]')

        if self.pps.mask_bg_stars_circle:
            self.mess('Mask background (circle) [sa]')
            self.sa_bg_mask = make_bg_circ_mask_cube(self.sa_sub[0].shape, self.sa_workcat,
                                        skip, radius=self.pps.mask_bg_radius,
                                        nthreads=self.pps.nthreads) * self.sa_apt
        else:
            self.mess('Mask background (PSF) [sa]')
            self.sa_bg_mask = make_bg_psf_mask_cube(self.sa_sub[0].shape, self.starcat.psf_ids,
                                        self.starcat.psfs, self.sa_workcat,
                                        skip,
                                        krn_scl=self.pps.motion_step,
                                        krn_rad=self.pps.motion_nsteps,
                                        radius=self.pps.mask_bg_radius,
                                        level=self.pps.mask_bg_level,
                                        nthreads=self.pps.nthreads) * self.sa_apt
        if self.pps.save_bg_mask_cube:
            self.save_cube_fits('bg_mask_cube_sa.fits', 
                                np.array(self.sa_bg_mask, dtype='uint8'))



    def make_bg_mask_im(self, skip=[0]):
        """Produces mask over stars image cube of properly located background stars, to
        be subtracted from observations. skip is a list of stars to be skipped;
        typically the target itself (0), but also the secondary in case of
        a binary. Also excludes stars closer than pps.mask_bg_star_sep from target.
        """
        if not self.pps.bgstars or not self.pps.mask_bg_stars:
            self.mess('No background stars masked [im]')
            return

        for n in range(1, self.starcat.catsize):
            if self.starcat.xpos[n]**2 + self.starcat.ypos[n]**2 <= self.pps.mask_bg_star_sep**2:
                skip.append(n)
            else:
                break
        skip = np.unique(skip)

        self.mess(f'Mask background stars; skip {skip} [im]')

        if self.pps.mask_bg_stars_circle:
            self.mess('Mask background (circle) [im]')
            self.im_bg_mask = make_bg_circ_mask_cube(self.im_sub[0].shape, self.im_workcat,
                                        skip, radius=self.pps.mask_bg_radius,
                                        nthreads=self.pps.nthreads) * self.im_apt
        else:
            self.mess('Mask background (PSF) [im]')
            self.im_bg_mask = make_bg_psf_mask_cube(self.im_sub[0].shape, self.starcat.psf_ids,
                                        self.starcat.psfs, self.im_workcat,
                                        skip,
                                        krn_scl=self.pps.motion_step,
                                        krn_rad=self.pps.motion_nsteps,
                                        radius=self.pps.mask_bg_radius,
                                        level=self.pps.mask_bg_level,
                                        nthreads=self.pps.nthreads) * self.im_apt
        if self.pps.save_bg_mask_cube:
            self.save_cube_fits('bg_mask_cube_im.fits', 
                                np.array(self.im_bg_mask, dtype='uint8'))


    def refine_star_bg_sa(self):
        """Fit PSF for the brightest stars to refine the subtraction, and
        also measure both photometry and astrometry of background stars.
        """
        self.mess('Refining star background model [sa]')
        lim_flux = self.pps.lim_fit / self.sa_nexp**.5
        inrad = self.pps.bg_star_inrad
        outrad = int(0.5*np.max(self.sa_debias[0].shape) +
                     0.25*psf_radii(lim_flux))
        starids = self.starcat.bright_star_ids(limflux=lim_flux,
                                               outradius=outrad,
                                               inradius=inrad)
        self.mess('Found {:d} stars > {:.1e} to model'.format(len(starids), lim_flux))
        modelcube = self.bg_model_sa() + self.sa_psf_cube
        self.sa_workcat = refine_star_bg(starids, self.sa_sub, self.sa_noise,
                                  self.sa_mask_cube, modelcube, self.sa_norm,
                                  work_cats=self.sa_workcat, 
                                  psf_ids=self.starcat.psf_ids,
                                  psfs=self.starcat.psfs,
                                  krn_scl=self.pps.motion_step,
                                  krn_rad=self.pps.motion_nsteps,
                                  nthreads=self.nthreads_reduced)
        if len(starids) > 0 and self.pps.save_bg_star_phot:
            gaiaID = [self.starcat.gaiaID[starid] for starid in starids]
            fluxes = np.zeros((len(self.sa_workcat), len(starids)))
            for n in range(len(self.sa_workcat)):
                fluxes[n] = self.sa_workcat[n].fscale[starids]
            filename = os.path.join(self.pps.outdir, 'bg_star_photometry_sa.fits')
            save_bg_star_phot_fits(filename,
                                self.sa_att[:,0],
                                self.mjd2bjd(self.sa_att[:,0]),
                                fluxes,
                                gaiaID,
                                self.sa_hdr)


    def refine_star_bg_im(self):
        """Fit PSF for the brightest stars to refine the subtraction, and
        also measure both photometry and astrometry of background stars.
        """
        self.mess('Refining star background model [im]')
        lim_flux = self.pps.lim_fit / self.im_nexp**.5
        inrad = self.pps.bg_star_inrad
        outrad = int(0.5*np.max(self.im_debias[0].shape) +
                     0.25*psf_radii(lim_flux))
        starids = self.starcat.bright_star_ids(limflux=lim_flux,
                                               outradius=outrad,
                                               inradius=inrad)
        self.mess('Found {:d} stars > {:.1e} to model'.format(len(starids), lim_flux))
        modelcube = self.bg_model_im() + self.im_psf_cube
        self.im_workcat = refine_star_bg(starids, self.im_sub, self.im_noise,
                                  self.im_mask_cube, modelcube, self.im_norm,
                                  work_cats=self.im_workcat, 
                                  psf_ids=self.starcat.psf_ids,
                                  psfs=self.starcat.psfs,
                                  krn_scl=self.pps.motion_step,
                                  krn_rad=self.pps.motion_nsteps,
                                  nthreads=self.nthreads_reduced)
        if len(starids) > 0 and self.pps.save_bg_star_phot:
            gaiaID = [self.starcat.gaiaID[starid] for starid in starids]
            fluxes = np.zeros((len(self.im_workcat), len(starids)))
            for n in range(len(self.im_workcat)):
                fluxes[n] = self.im_workcat[n].fscale[starids]
            filename = os.path.join(self.pps.outdir, 'bg_star_photometry_im.fits')
            save_bg_star_phot_fits(filename,
                                self.im_att[:,0],
                                self.mjd2bjd(self.im_att[:,0]),
                                fluxes,
                                gaiaID,
                                self.im_hdr)


    def has_source(self, data, mask=None, clip=5, niter=10):
        """Find out what frames are missing a source, and return
        a binary index array to indicate frames with source.
        """
        self.mess('Finding frames with source missing')
        contrast = rough_contrast(data, mask=mask)
        sel, ret_str = check_low(contrast, clip, niter)
        self.mess(ret_str, 2)
        return sel


    def filter_flux(self, flux, clip=5, niter=10):
        """Verifies the consistency of the photometry and de-selects
        data points with too deviating flux. Returns a binary index array
        of consistent data points.
        """
        self.mess('Filtering out frames with too deviating summed flux')
        sel, ret_str = check_val(flux, clip, niter)
        self.mess(ret_str, 2)
        return sel

    
    def filter_pos(self, xc, yc, clip=5, niter=3):
        """Returns a binary index to positions without
        the outliers.
        """
        self.mess('Finding frames with excessive offsets')
        sel, ret_str = check_pos(xc, yc, clip, niter)
        self.mess(ret_str, 2)
        return sel
    
    
    def filter_motion(self, xc, yc, lowfrac=0.9):
        """Selects frames with positions after and before that are fairly 
        consistent, with the goal to select frames with little 
        motion blur. lowfrac is the fraction of frames to keep,
        per coordinate.
        """
        self.mess('Filtering out frames with suspected motion')
        selX, ret_str = check_motion(xc, lowfrac)
        self.mess(ret_str, 2)
        selY, ret_str =  check_motion(yc, lowfrac)
        self.mess(ret_str, 2)
        return selX*selY
    
    
    def filter_bad_masks(self, mask_cube, apt, clip=5, niter=3):
        """Returns binary index of frames without excessively number
        of masked pixels (that could be the result of, e.g., a satellite
        crossing the field of view, or some strongly scattered light)
        """
        self.mess('Finding frames with excessively masked pixels')
        sel, ret_str = check_mask(mask_cube, apt, clip, niter)
        self.mess(ret_str, 2)
        return sel
    
        
    def filter_high_bg(self, bg, clip=5, niter=10, sel=None):
        """Returns binary index of frames without too deviating background,
        as often happens when pointing is close to Earth limb
        """
        self.mess('Finding frames with deviating background')
        if sel is None:
            sel = np.ones(len(bg), dtype='?')
        bg_sel, ret_str = check_val(bg[sel], clip=clip, niter=niter)
        self.mess(ret_str, 2)
        ret_sel = np.ones(len(bg), dtype='?')
        ret_sel[sel] = bg_sel
        return ret_sel
    
    
    def sa2im_crop(self, data):
        """Crop out the imagette region from a subarray. Works for
        both frames and cubes.
        """
        i0 = int(self.im_sa_off[0])
        i1 = int(i0+self.im_debias[0].shape[1])
        j0 = int(self.im_sa_off[1])
        j1 = int(j0+self.im_debias[0].shape[0])

        if data.ndim == 2:
            return data[j0:j1, i0:i1]
        return data[:, j0:j1, i0:i1]
   
    
    def chi2_sa(self, psf_model):        
        """Compute the chi2 for how well the psf_model fit to data.
        """
        qc2 = ((self.sa_sub-psf_model)/self.sa_noise)**2
        chi2 = np.zeros(len(qc2))
        for n in range(len(qc2)):
            chi2[n] = np.mean(qc2[n][self.sa_mask_cube[n]])
        return chi2
    
    
    def chi2_im(self, psf_model):
        """Compute the chi2 for how well the psf_model fit to data.
        """
        qc2 = ((self.im_sub-psf_model)/self.im_noise)**2
        chi2 = np.zeros(len(qc2))
        for n in range(len(qc2)):
            chi2[n] = np.mean(qc2[n][self.im_mask_cube[n]])
        return chi2


    def read_Teff(self, entry=0):
        """The effective temperature of the target is typically defined in
        the starcat file, so this method retrieves and returns it.
        """
        self.mess('Reading T_EFF from starcat file ')
        temp = starcat(self.pps.file_starcat, colstr='T_EFF', entry=entry)
        self.mess(' Target T_eff = {:.1f}'.format(temp))
        return temp
    

    def save_eigen_sa(self, flag, flux, err, bg, w):
        """Save results from the photometric extraction into a 
        fits file located in the output directory.
        """
        filename = os.path.join(self.pps.outdir, f'{self.pps.name}_{self.pps.visit}_sa.fits')
        save_eigen_fits(filename,
                          self.sa_att[:,0],
                          self.mjd2bjd(self.sa_att[:,0]),
                          flux,
                          err,
                          bg,
                          self.sa_att[:,3],
                          self.sa_xc,
                          self.sa_yc,
                          flag,
                          w,
                          self.sa_thermFront_2,
                          self.sa_hdr)


    def save_eigen_im(self, flag, flux, err, bg, w):
        """Save results from the photometric extraction into a 
        fits file located in the output directory.
        """
        filename = os.path.join(self.pps.outdir, f'{self.pps.name}_{self.pps.visit}_im.fits')
        save_eigen_fits(filename,
                          self.im_att[:,0],
                          self.mjd2bjd(self.im_att[:,0]),
                          flux,
                          err,
                          bg,
                          self.im_att[:,3],
                          self.im_xc,
                          self.im_yc,
                          flag,
                          w,
                          self.im_thermFront_2,
                          self.im_hdr)


    def save_cube_fits(self, filename, cube):
        """Save a data cube as a fits file.
        """
        save_cube_fits(os.path.join(self.pps.outdir, filename), cube)


    def compute_residuals_sa(self):
        """Subtract model from data and return cube
        of residuals
        """
        model = self.sa_psf_cube + self.bg_model_sa()
        res = self.sa_sub - model
        return res


    def compute_residuals_im(self):
        """Subtract model from data and return cube
        of residuals
        """
        model = self.im_psf_cube + self.bg_model_im()
        res = self.im_sub - model
        return res


    def save_residuals_sa(self, prefix):
        """Subtract model including bg stars from data and 
        save as a fits cube in the output directory.
        """
        res = self.compute_residuals_sa()
        self.save_cube_fits(prefix+'residuals_sa.fits', res)


    def save_residuals_im(self, prefix):
        """Subtract model from data and save as a fits cube
        in the output directory.
        """
        res = self.compute_residuals_im()
        self.save_cube_fits(prefix+'residuals_im.fits', res)    


    def save_bg_sa(self, prefix):
        """Subtract model (but not bg stars) from data and save 
        as a fits cube in the output directory.
        """
        res = self.compute_residuals_sa()
        res += self.bg_model_sa()
        self.save_cube_fits(prefix+'bgresiduals_sa.fits', res)


    def save_bg_im(self, prefix):
        """Subtract target PSF model (but not bg stars) from data and save 
        as a fits cube in the output directory.
        """
        res = self.compute_residuals_im()
        res += self.bg_model_im()
        self.save_cube_fits(prefix+'bgresiduals_im.fits', res)


    def save_bg_model_sa(self, prefix):
        """Save bg model including bg stars, smear, and static
        """
        self.save_cube_fits(prefix+'bg_model_sa.fits', self.bg_model_sa())


    def save_bg_model_im(self, prefix):
        """Save bg model including bg stars, smear, and static
        """
        self.save_cube_fits(prefix+'bg_model_im.fits', self.bg_model_im())


    #----------- Methods for binary extractions below

    def define_binary_psf_library(self):
        """Loads PSF eigen library for both components. If second PSF
        library is not defined, the first is used.
        """
        if self.pps.psflib0 is None:
            if self.pps.psflib is not None:
                self.pps.psflib0 = self.pps.psflib
            else:
                self.pps.psflib0 = 0            
        if isinstance(self.pps.psflib0, str):
            file_psf0 = self.pps.psflib0
        else:
            file_psf0 = self.eigen_name(self.pps.psflib0)
        if self.pps.psflib1 is None:
            file_psf1 = file_psf0
        else:
            if isinstance(self.pps.psflib1, str):
                file_psf1 = self.pps.psflib1
            else:
                file_psf1 = self.eigen_name(self.pps.psflib1)
        if not os.path.isfile(file_psf0):
            raise Exception(f'PSF eigen library file A \"{file_psf0}\" not found')
        if not os.path.isfile(file_psf1):
            raise Exception(f'PSF eigen library file B \"{file_psf1}\" not found')

        self.eigen_psf0 = pickle.load(open(file_psf0, 'rb'))
        self.eigen_psf1 = pickle.load(open(file_psf1, 'rb'))
        self.eigen_psf = self.eigen_psf0
        self.psf0 = self.eigen_psf0[0]
        self.psf1 = self.eigen_psf1[0]
        self.psf = self.psf0

        self.mess('Binary PSF eigen libraries defined ')
        self.mess(f' A: \"{file_psf0}\"')
        self.mess(f' B: \"{file_psf1}\"')

        if self.pps.bg_psflib is None:
            self.bg_psf = self.psf
        else:
            if isinstance(self.pps.bg_psflib, str):
                file_psf = self.pps.bg_psflib
            else:
                file_psf = self.eigen_name(self.pps.bg_psflib)
            if not os.path.isfile(file_psf):
                raise Exception(f'BG PSF eigen library file \"{file_psf}\" not found')

            self.mess('BG PSF eigen library defined: ')
            self.mess(f' \"{file_psf}\"')
            self.bg_psf = pickle.load(open(file_psf, 'rb'))[0]


    def pre_binary(self, integ_rad=15):
        """Special pre-processing for a binary. integ_rad is the integration 
        radius to be used for normalisation of PSFs and depends a bit on the
        separation. Can be tricky for strongly unresolved binaries, but is
        only an initial guess anyway so not critical.
        """
        self.pre_process()
        psf_flux0 = integrate_psf(self.psf0, radius=integ_rad)
        psf_flux1 = integrate_psf(self.psf1, radius=integ_rad)
        sa_med_flux = self.median_flux(self.sa_sub, self.sa_mask, radius=integ_rad)
        self.sa_norm0 = sa_med_flux/psf_flux0
        self.sa_norm1 = sa_med_flux/psf_flux1
        self.mess('Binary normalisation: {:.4e}, {:.4e}'.format(self.sa_norm0, 
                                                                self.sa_norm1))
        self.sa_psf0 = lambda y, x, grid=True : self.sa_norm0 * self.psf0(y, x, grid=grid)
        self.sa_psf1 = lambda y, x, grid=True : self.sa_norm1 * self.psf1(y, x, grid=grid)
        if self.pps.init_flux_ratio is None:
            self.pps.init_flux_ratio = self.starcat.fscale[self.pps.secondary]


    def process_binary_sa(self):
        """
        """
        self.mess('--- Start processing subarray binary data with eigen')
        klip = self.pps.klip
        if klip is None:
            klip = len(self.eigen_psf)
        else:
            klip = min(klip, len(self.eigen_psf))
        sel = self.sa_cent_sel
        
        if self.pps.centre:
            self.centre_binary_deconv_sa(self.centre_psf)
#            if self.pps.robust_centre_binary:
#                self.robust_centre_binary_sa(self.psf)
#            else:
#                self.centre_binary_sa(self.psf)
        else:
            self.define_binary_coordinates_sa()

        fix_flux2 = None
        for n in range(self.pps.sigma_clip_niter):
            self.mess('--- Iteration binary sa {:d}/{:d}'.format(n+1, self.pps.sigma_clip_niter))

            psf_cube00, psf_cube10, scale00, scale10, bg0, w00, w10 = multi_psf_fit_binary(
                            self.eigen_psf0[:klip],
                            self.eigen_psf1[:klip],
                            self.sa_sub[sel] - self.sa_stat_res,
                            self.sa_noise[sel],
                            self.sa_mask_cube[sel],
                            self.sa_xc0[sel], self.sa_yc0[sel],
                            self.sa_xc1[sel], self.sa_yc1[sel],
                            psfrad=self.pps.sa_psfrad, fitrad=self.pps.fitrad,
                            krn_scl=self.pps.motion_step, krn_rad=self.pps.motion_nsteps,
                            nthreads=self.pps.nthreads, fix_flux2=fix_flux2, 
                            non_negative=self.pps.non_neg_lsq)

            psf_cube00 *= self.sa_apt
            psf_cube10 *= self.sa_apt
            t0 = self.sa_att[sel, 0]
            t = self.sa_att[:, 0]        
            psf_cube0 = interp_cube_ext(t, t0, psf_cube00)
            psf_cube1 = interp_cube_ext(t, t0, psf_cube10)
            w0 = interp_cube_ext(t, t0, w00)
            w1 = interp_cube_ext(t, t0, w10)
            scale0 = np.interp(t, t0, scale00)
            scale1 = np.interp(t, t0, scale10)
            if self.pps.fix_flux2:
                fix_flux2 = np.nanmedian(scale10)
            self.mess('Iter {:d} MAD sa0: {:.2f} ppm'.format(n+1, mad(scale00)))
            self.mess('Iter {:d} MAD sa1: {:.2f} ppm'.format(n+1, mad(scale10)))
            bg = np.interp(t, t0, bg0)
            self.make_mask_cube_sa(psf_cube0+psf_cube1, bg)
            self.sa_mask_cube[sel==0] = self.sa_mask
            res = self.sa_sub - (psf_cube0+psf_cube1) - bg[:,None,None]
            if self.pps.remove_static:
                self.sa_stat_res = np.nanmedian(res, axis=0)
#            self.remove_resid_smear_sa(res, (psf_cube0+psf_cube1))
            self.sa_noise = (self.psf_noise_sa((psf_cube0+psf_cube1) + self.sa_bg[:, None, None] +
                bg[:, None, None] + 2*self.sa_bgstars + np.abs(self.sa_smear[:, None, :]) + 
                np.abs(self.sa_stat_res) + np.abs(self.sa_dark))**2 +
                self.sa_dark_err**2)**.5

        if self.pps.save_resid_cube:
            self.save_residuals_sa('binary_', (psf_cube0+psf_cube1), bg)
        
        flagCenter = (self.filter_pos(self.sa_xc0, self.sa_yc0) == 0)
        flagBadPix = (self.filter_bad_masks(self.sa_mask_cube, self.sa_apt) == 0)
        flagFlux = (self.filter_flux(scale0 + scale1) == 0)
        flagSource = (sel == 0)
        flagBG = (self.filter_high_bg(bg + self.sa_bg, sel=(flagFlux==0)) == 0)
        flag = np.zeros(flagCenter.shape, dtype='int8')
        flag[:] = (1*flagCenter + 2*flagBadPix + 4*flagFlux +
                    8*flagSource + 16*flagBG)
        self.save_binary_eigen_sa(flag, scale0, scale1,
                           bg + self.sa_bg, w0, w1)
        if self.pps.save_mask_cube:
            self.save_cube_fits('mask_cube_sa.fits', 
                                np.array(self.sa_mask_cube, dtype='uint8'))
        if self.pps.save_psfmodel:
            self.save_cube_fits('psf_model0_sa.fits', psf_cube0)
            self.save_cube_fits('psf_model1_sa.fits', psf_cube1)

        self.mess('MAD sa0: {:.2f} ppm'.format(mad(scale0)))
        self.mess('MAD sa0[flag==0]: {:.2f} ppm'.format(mad(scale0[flag==0])))
        self.mess('MAD sa1: {:.2f} ppm'.format(mad(scale1)))
        self.mess('MAD sa1[flag==0]: {:.2f} ppm'.format(mad(scale1[flag==0])))
        return psf_cube0, psf_cube1, bg + self.sa_bg


    def process_binary_im(self):
        self.mess('--- Start processing imagette binary data with eigen')
        klip = self.pps.klip
        if klip is None:
            klip = len(self.eigen_psf)
        else:
            klip = min(klip, len(self.eigen_psf))
        sel = self.im_cent_sel

        if self.pps.centre:
            self.centre_binary_deconv_im(self.centre_psf)
#            if self.pps.robust_centre_binary:
#                self.robust_centre_binary_im(self.psf)
#            else:
#                self.centre_binary_im(self.psf)
        else:
            self.define_binary_coordinates_im()
        fix_flux2 = None
       
        for n in range(self.pps.sigma_clip_niter):
            self.mess('--- Iteration binary im {:d}/{:d}'.format(n+1, self.pps.sigma_clip_niter))

            psf_cube00, psf_cube10, scale00, scale10, bg0, w00, w10 = multi_psf_fit_binary(
                            self.eigen_psf0[:klip],
                            self.eigen_psf1[:klip],
                            self.im_sub[sel] - self.im_stat_res,
                            self.im_noise[sel],
                            self.im_mask_cube[sel],
                            self.im_xc0[sel], self.im_yc0[sel],
                            self.im_xc1[sel], self.im_yc1[sel],
                            psfrad=self.pps.sa_psfrad, fitrad=self.pps.fitrad,
                            krn_scl=self.pps.motion_step, krn_rad=self.pps.motion_nsteps,
                            nthreads=self.pps.nthreads, fix_flux2=fix_flux2,
                            non_negative=self.pps.non_neg_lsq)

            psf_cube00 *= self.im_apt
            psf_cube10 *= self.im_apt
            t0 = self.im_att[sel, 0]
            t = self.im_att[:, 0]        
            psf_cube0 = interp_cube_ext(t, t0, psf_cube00)
            psf_cube1 = interp_cube_ext(t, t0, psf_cube10)
            w0 = interp_cube_ext(t, t0, w00)
            w1 = interp_cube_ext(t, t0, w10)
            scale0 = np.interp(t, t0, scale00)
            scale1 = np.interp(t, t0, scale10)
            if self.pps.fix_flux2:
                fix_flux2 = np.nanmedian(scale10)
            self.mess('Iter {:d} MAD im0: {:.2f} ppm'.format(n+1, mad(scale00)))
            self.mess('Iter {:d} MAD im1: {:.2f} ppm'.format(n+1, mad(scale10)))
            bg = np.interp(t, t0, bg0)
            self.make_mask_cube_im(psf_cube0+psf_cube1, bg)
            self.im_mask_cube[sel==0] = self.im_mask
            res = self.im_sub - (psf_cube0+psf_cube1) - bg[:,None,None]
            if self.pps.remove_static:
                self.im_stat_res = np.nanmedian(res, axis=0)
            self.im_noise = (self.psf_noise_im((psf_cube0+psf_cube1) + self.im_bg[:, None, None] +
                bg[:, None, None] + 2*self.im_bgstars + np.abs(self.im_smear[:, None, :]) + 
                np.abs(self.im_stat_res) + np.abs(self.im_dark))**2 +
                self.im_dark_err**2)**.5

        if self.pps.save_resid_cube:
            self.save_residuals_im('binary_', (psf_cube0+psf_cube1), bg)
        
        flagCenter = (self.filter_pos(self.im_xc0, self.im_yc0) == 0)
        flagBadPix = (self.filter_bad_masks(self.im_mask_cube, self.im_apt) == 0)
        flagFlux = (self.filter_flux(scale0 + scale1) == 0)
        flagSource = (sel == 0)
        flagBG = (self.filter_high_bg(bg + self.im_bg, sel=(flagFlux==0)) == 0)
        flag = np.zeros(flagCenter.shape, dtype='int8')
        flag[:] = (1*flagCenter + 2*flagBadPix + 4*flagFlux +
                    8*flagSource + 16*flagBG)
        self.save_binary_eigen_im(flag, scale0, scale1,
                           bg + self.im_bg, w0, w1)
        if self.pps.save_mask_cube:
            self.save_cube_fits('mask_cube_im.fits', 
                                np.array(self.im_mask_cube, dtype='uint8'))
        if self.pps.save_psfmodel:
            self.save_cube_fits('psf_model0_im.fits', psf_cube0)
            self.save_cube_fits('psf_model1_im.fits', psf_cube1)
        self.mess('MAD im0: {:.2f} ppm'.format(mad(scale0)))
        self.mess('MAD im0[flag==0]: {:.2f} ppm'.format(mad(scale0[flag==0])))
        self.mess('MAD im1: {:.2f} ppm'.format(mad(scale1)))
        self.mess('MAD im1[flag==0]: {:.2f} ppm'.format(mad(scale1[flag==0])))
        return psf_cube0, psf_cube1,  bg + self.im_bg

    def centre_binary_deconv_sa(self, psf):
        dx, dy = self.starcat.rotate_entry(self.pps.secondary, self.sa_att[:,3])
        xi = int(0.5*self.sa_sub[0].shape[1])
        yi = int(0.5*self.sa_sub[0].shape[0])

        self.mess('Compute subarray deconvolved centers [sa] (multi {:d} threads)'.format(self.pps.nthreads))
        self.sa_xc0, self.sa_yc0, self.sa_xc1, self.sa_yc1 = (
                multi_cent_binary_deconvolve(psf,
                                            self.sa_sub,
                                            dx, dy, 
                                            xi, yi, 
                                            radius=self.pps.centfit_rad,
                                            subrad=3,
                                            nthreads=self.pps.nthreads))
            
        ds = ((self.sa_xc1-self.sa_xc0)**2 + (self.sa_yc1-self.sa_yc0)**2)**.5

        if self.pps.save_astrometry:
            filename = os.path.join(self.pps.outdir, f'{self.pps.name}_{self.pps.visit}_separation.txt')
            np.savetxt(filename, ds)

        self.separation = np.median(ds)
        self.mess('Astrometry: separation [sa] = {:.3f} +/- {:.3f} pix'.format(
                self.separation, np.std(ds)/len(ds)**.5))


    def centre_binary_sa(self, psf):
        dx, dy = self.starcat.rotate_entry(self.pps.secondary, self.sa_att[:,3])
        xc1, yc1 = self.sa_xc + dx, self.sa_yc + dy
        norm0 = self.sa_norm0
        norm1 = self.pps.init_flux_ratio*self.sa_norm1
        
        self.mess('Compute subarray PSF centers [sa] (multi {:d} threads)'.format(self.pps.nthreads))
        sa_xc0, sa_yc0, _sc0, sa_xc1, sa_yc1, _sc1 = (
                multi_cent_binary_psf(psf,
                                      self.sa_sub, self.sa_noise,
                                      self.sa_xc, self.sa_yc,
                                      xc1, yc1,
                                      norm0, norm1,
                                      self.sa_mask_cube,
                                      radius=self.pps.centfit_rad,
                                      nthreads=self.pps.nthreads))
            
        x, y = derotate_position(sa_xc1-sa_xc0, sa_yc1-sa_yc0, self.sa_att[:,3])
        self.binary_x1, self.binary_y1 = np.median(x), np.median(y)
        ds = (x**2+y**2)**.5

        if self.pps.save_astrometry:
            filename = os.path.join(self.pps.outdir, f'{self.pps.name}_{self.pps.visit}_separation.txt')
            np.savetxt(filename, ds)

        self.separation = np.median(ds)
        self.mess('Astrometry: separation [sa] = {:.3f} +/- {:.3f} pix'.format(
                self.separation, np.std(ds)/len(ds)**.5))
        
        dx, dy = rotate_position(self.binary_x1, self.binary_y1, self.sa_att[:,3])
        
        self.mess('Compute subarray PSF centers [sa] (multi {:d} threads)'.format(self.pps.nthreads))
        self.sa_xc0, self.sa_yc0, _sc0, self.sa_xc1, self.sa_yc1, _sc1 = (
                multi_cent_binary_psf_fix(psf,
                                          self.sa_sub, self.sa_noise,
                                          self.sa_xc, self.sa_yc, 
                                          dx, dy,
                                          norm0, norm1, 
                                          self.sa_mask_cube,
                                          radius=self.pps.centfit_rad,
                                          nthreads=self.pps.nthreads))


    def centre_binary_im(self, psf):
        dx, dy = self.starcat.rotate_entry(self.pps.secondary, self.im_att[:,3])
        xc1, yc1 = self.im_xc + dx, self.im_yc + dy
        norm0 = self.im_norm0
        norm1 = self.pps.init_flux_ratio*self.im_norm1
        
        self.mess('Compute subarray PSF centers [im] (multi {:d} threads)'.format(self.pps.nthreads))
        im_xc0, im_yc0, _sc0, im_xc1, im_yc1, _sc1 = (
                multi_cent_binary_psf(psf,
                                      self.im_sub, self.im_noise,
                                      self.im_xc, self.im_yc,
                                      xc1, yc1,
                                      norm0, norm1,
                                      self.im_mask_cube,
                                      radius=self.pps.centfit_rad,
                                      nthreads=self.pps.nthreads))        
            
        x, y = derotate_position(im_xc1-im_xc0, im_yc1-im_yc0, self.im_att[:,3])
        self.binary_x1, self.binary_y1 = np.median(x), np.median(y)
        ds = (x**2+y**2)**.5

        if self.pps.save_astrometry:
            filename = os.path.join(self.pps.outdir, f'{self.pps.name}_{self.pps.visit}_separation.txt')
            np.savetxt(filename, ds)

        self.separation = np.median(ds)
        self.mess('Astrometry [im]: separation = {:.3f} +/- {:.3f} pix'.format(
                self.separation, np.std(ds)/len(ds)**.5))
        
        dx, dy = rotate_position(self.binary_x1, self.binary_y1, self.im_att[:,3])
        
        self.mess('Compute imagette PSF centers [im] (multi {:d} threads)'.format(self.pps.nthreads))
        self.im_xc0, self.im_yc0, _sc0, self.im_xc1, self.im_yc1, _sc1 = (
                multi_cent_binary_psf_fix(psf,
                                          self.im_sub, self.im_noise,
                                          self.im_xc, self.im_yc, 
                                          dx, dy,
                                          norm0, norm1, 
                                          self.im_mask_cube,
                                          radius=self.pps.centfit_rad,
                                          nthreads=self.pps.nthreads))    

    def centre_binary_deconv_im(self, psf):
        dx, dy = self.starcat.rotate_entry(self.pps.secondary, self.im_att[:,3])
        xi = int(0.5*self.im_sub[0].shape[1])
        yi = int(0.5*self.im_sub[0].shape[0])

        self.mess('Compute subarray deconvolved centers [im] (multi {:d} threads)'.format(self.pps.nthreads))
        self.im_xc0, self.im_yc0, self.im_xc1, self.im_yc1 = (
                multi_cent_binary_deconvolve(psf,
                                            self.im_sub,
                                            dx, dy, 
                                            xi, yi, 
                                            radius=self.pps.centfit_rad,
                                            subrad=3,
                                            nthreads=self.pps.nthreads))
            
        ds = ((self.im_xc1-self.im_xc0)**2 + (self.im_yc1-self.im_yc0)**2)**.5

        if self.pps.save_astrometry:
            filename = os.path.join(self.pps.outdir, f'{self.pps.name}_{self.pps.visit}_separation.txt')
            np.savetxt(filename, ds)

        self.separation = np.median(ds)
        self.mess('Astrometry: separation [im] = {:.3f} +/- {:.3f} pix'.format(
                self.separation, np.std(ds)/len(ds)**.5))





    def robust_centre_binary_sa(self, psf):
        xc0 = np.median(self.sa_xc)*np.ones_like(self.sa_xc)
        yc0 = np.median(self.sa_yc)*np.ones_like(self.sa_yc)
        dx, dy = self.starcat.rotate_entry(self.pps.secondary, self.sa_att[:,3])

        norm0 = self.sa_norm0
        norm1 = self.pps.init_flux_ratio*self.sa_norm1
        
        self.mess('Compute robust subarray PSF centers (multi {:d} threads)'.format(self.pps.nthreads))
        self.sa_xc0, self.sa_yc0, _sc0, self.sa_xc1, self.sa_yc1, _sc1 = (
                multi_cent_binary_psf_fix(psf,
                                          self.sa_sub, self.sa_noise,
                                          xc0, yc0, 
                                          dx, dy,
                                          norm0, norm1, 
                                          self.sa_mask_cube,
                                          radius=self.pps.centfit_rad,
                                          nthreads=self.pps.nthreads))        

    def robust_centre_binary_im(self, psf):
        xc0 = np.median(self.im_xc)*np.ones_like(self.im_xc)
        yc0 = np.median(self.im_yc)*np.ones_like(self.im_yc)
        dx, dy = self.starcat.rotate_entry(self.pps.secondary, self.im_att[:,3])

        norm0 = self.im_norm0
        norm1 = self.pps.init_flux_ratio*self.im_norm1
        
        self.mess('Compute robust subarray PSF centers (multi {:d} threads)'.format(self.pps.nthreads))
        self.im_xc0, self.im_yc0, _sc0, self.im_xc1, self.im_yc1, _sc1 = (
                multi_cent_binary_psf_fix(psf,
                                          self.im_sub, self.im_noise,
                                          xc0, yc0, 
                                          dx, dy,
                                          norm0, norm1, 
                                          self.im_mask_cube,
                                          radius=self.pps.centfit_rad,
                                          nthreads=self.pps.nthreads))        


    def define_binary_coordinates_sa(self):
        self.sa_xc0 = self.sa_xc
        self.sa_yc0 = self.sa_yc
        dx, dy = self.starcat.rotate_entry(self.pps.secondary, self.sa_att[:,3])        
        self.sa_xc1 = self.sa_xc0 + dx
        self.sa_yc1 = self.sa_yc0 + dy

    def define_binary_coordinates_im(self):
        self.im_xc0 = self.im_xc
        self.im_yc0 = self.im_yc
        dx, dy = self.starcat.rotate_entry(self.pps.secondary, self.im_att[:,3])        
        self.im_xc1 = self.im_xc0 + dx
        self.im_yc1 = self.im_yc0 + dy


    def save_binary_eigen_sa(self, flag, flux0, flux1, bg, w0, w1):
        """Save results from the photometric extraction of a binary into a 
        fits file located in the output directory.
        """
        filename = os.path.join(self.pps.outdir, f'{self.pps.name}_{self.pps.visit}_binary_sa.fits')
        save_binary_eigen_fits(filename,
                          self.sa_att[:,0],
                          self.mjd2bjd(self.sa_att[:,0]),
                          flux0,
                          flux1,
                          bg,
                          self.sa_att[:,3],
                          self.sa_xc0,
                          self.sa_yc0,
                          self.sa_xc1,
                          self.sa_yc1,
                          flag,
                          w0,
                          w1,
                          self.sa_thermFront_2,
                          self.sa_hdr)


    def save_binary_eigen_im(self, flag, flux0, flux1, bg, w0, w1):
        """Save results from the photometric extraction of a binary into a 
        fits file located in the output directory.
        """
        filename = os.path.join(self.pps.outdir, f'{self.pps.name}_{self.pps.visit}_binary_im.fits')
        save_binary_eigen_fits(filename,
                          self.im_att[:,0],
                          self.mjd2bjd(self.im_att[:,0]),
                          flux0,
                          flux1,
                          bg,
                          self.im_att[:,3],
                          self.im_xc0,
                          self.im_yc0,
                          self.im_xc1,
                          self.im_yc1,
                          flag,
                          w0,
                          w1,
                          self.im_thermFront_2,
                          self.im_hdr)


if __name__=='__main__':
    print('Do not run')
    
