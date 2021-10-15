# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 00:27:24 2020

@author: Alexis Brandeker, alexis@astro.su.se

Master class PsfPhot of PIPE contains the data and routines to
manipulate the and extract PSF photometry. Has separate but 
similar paths for subarray [sa] and imagette [im] manipulation,
and also for entangled binaries where two componens are fit 
simultaneously.

Most of the detailed functionality is provided by other modules
that are called from PsfPhot.

"""
import os
import pickle
import numpy as np
from scipy.ndimage import shift
from astropy.io import fits

from .analyse import mad, psf_phot_cube, sigma_clip, smo_spl_bg
from .cent import (
    flux as cent_flux
)
from .multi_cent import (
    psf as multi_cent_psf,
    binary_psf as multi_cent_binary_psf,
    binary_psf_fix as multi_cent_binary_psf_fix
)
from .multi_psf import fit as multi_psf_fit, fit_binary as multi_psf_fit_binary
from .pipe_log import PipeLog
from .read import (
    imagette_offset, datacube, mask as read_mask, attitude, gain as read_gain,
    ron as read_ron, thermFront_2, mjd2bjd, nonlinear, flatfield, starcat,
    save_eigen_fits, save_binary_eigen_fits, sub_image_indices,
    dark as read_dark
)
from .syntstar import star_bg, rotate_position, derotate_position
from .reduce import (
    resample_attitude, resample_imagette_time, aperture, integrate_psf,
    interp_cube_ext, cube_apt, clean_cube2D, interp_cube, noise, psf_noise,
    pix_mat, make_maskcube, rough_contrast, check_low, check_val, check_pos,
    check_motion, check_mask, empiric_noise
)


class PsfPhot:
    """ Reads and stores CHEOPS data and collects methods to analyse
        the data using mainly PSF photometry.
    """
    def __init__(self, pipe_params):
        self.pps = pipe_params
        
        self.plog = PipeLog(self.pps.file_log, self.pps.plog_verbosity)
        self.mess = self.plog.mess  # Shorthand
        
        self.plog.mess_list(self.pps.str_list()) # Save list of params to log
        
        if self.pps.Teff is None:
            self.pps.Teff = self.read_Teff()
        
        # ----------- General variables
        self.eigen_psf = None   # Library of PSF eigen components
        self.psf = None         # Default PSF to be used when not
                                # fitting eigen components
        self.starcat = None     # Star Catalogue object containing bg stars 
        self.mjd2bjd = None     # Defines a function to be used to convert
                                # between MJD and BJD
        # ----------- Subarray variables
        self.sa_psf = None      # Default PSF normalised to subarray flux
        self.sa_debias = None   # Cube of DRP calibrated subarray frames
                                # (bias removed, flatfielded, nonlinear corrected)
        self.sa_sub = None      # Reduced cube with background (and dark)
                                # subtracted, smearing corrected
        self.sa_bgstars = None  # Cube of background stars in subarray
        self.sa_dark = None     # Dark current subarray image scaled to exp. time
        self.sa_dark_err = None # Std error of subarray dark current image
        self.sa_mask = None     # Mask image for bad pixels in subarray
        self.sa_apt = None      # Image that defines active pixels in cube
                                # (e.g. with circular aperture not all pixels)
        self.sa_bg = None       # Background value, a constant for each frame
                                # in cube list
        self.sa_ring95 = None   # Part of defined image outside radius 95 pixels
                                # - used to correct for systematic bg subtraction
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
        
        # ----------- Imagette variables
        self.nexp = None        # subarray/imagette relative exposure time
        self.im_psf = None      # Default PSF normalised to imagette flux
        self.im_gain = None     # Array of gain in e/ADU for each epoch
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
        self.im_xc = None       # Array of PSF x-coordinate centres for imagettes
        self.im_yc = None       # Array of PSF y-coordinate centres for imagettes
        self.im_smear = None    # Smearing correction, 1D vector for each frame
        self.im_stat_res = None # Median of residuals, a derived static image
                                # Can be used as a dark current image
        self.im_thermFront_2 = None  # thermFront_2 vector, passed on to save-file

        self.mad_sa = None
        self.mad_im = None
        # Read and initialise data from files
        self.read_data()

        # Define gain and read-out noise        
        self.read_gain()
        self.read_ron()


    def read_data(self):
        """Initialises data from data files
        """
        # Read subarray data cube and define
        # sa_debias, sa_mjd, sa_hdr, sa_tab
        self.read_sa_datacube()
        
        # Read and define the PSF library. Allow for two different PSF
        # libraries in case of a binary
        if self.pps.binary:
            self.define_binary_psf_library()
        else:
            self.define_psf_library()

        self.sa_smear = np.zeros(self.sa_debias.shape[0:2])
        self.sa_stat_res = np.zeros(self.sa_debias[0].shape)

        # Check if imagettes are available, load data and define
        # corresponding parameters
        if self.pps.file_im is not None:
            # Derive imagette range from subarray range
            self.read_imagettes(im_range=(0,1))
            self.nexp = self.sa_hdr['NEXP']/self.im_hdr['NEXP']
            if self.pps.sa_range is not None:
                im_range = (int(self.pps.sa_range[0]*self.nexp), 
                           int(self.pps.sa_range[1]*self.nexp))
            else:
                im_range=None
            self.read_imagettes(im_range=im_range)
            # A bug in the DPS caused imagette timings to be wrong. This
            # can be easily fixed by resampling times from subarray timings
            if self.pps.resample_im_times:
                self.im_mjd = resample_imagette_time(self.sa_mjd, self.nexp)
            self.im_smear = np.zeros(self.im_raw.shape[0:2])
            self.im_stat_res = np.zeros(self.im_raw[0].shape)
        else:
            self.nexp = 1

        self.read_attitude()
        self.read_thermFront_2()
        self.read_mask()
        self.read_darks()
        self.define_mjd2bjd()
        self.read_starcat()


    def define_psf_library(self):
        """Loads PSF eigen library. If psf_name is a number, a
        matching filename is generated. If psf_name is a string,
        that us instead interpreted as an explicit filename (with
        full path).
        """
        if self.pps.psflib is None:
            self.pps.psflib = 0
        if isinstance(self.pps.psflib, str):
            file_psf = self.pps.psflib
        else:
            file_psf = self.eigen_name(self.pps.psflib)
        if not os.path.isfile(file_psf):
            raise Exception(f'PSF eigen library file \"{file_psf}\" not found')

        self.mess('PSF eigen library defined: ')
        self.mess(f' \"{file_psf}\"')

        self.eigen_psf = pickle.load(open(file_psf, 'rb'))
        self.psf = self.eigen_psf[0]

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




    def find_next_lib_num(self, name_fun, max_num=100000):
        """Find first available number without existing
        library file. If no available number is found,
        and exception is returned. name_fun is the function
        that generates a filename given a number. max_num
        is the largest number to be tried before giving up.
        """
        for lib_num in range(max_num):
            filename = name_fun(lib_num)
            if not os.path.isfile(filename):
                return lib_num
        raise Exception('No available PSF library numbers')


    def psf_name(self, lib_num, component='A'):
        """Returns full path name to psf library. Name is of the format
        psfs_{x}_{y}_{R}_{name}_{visit}_{lib_num}{component}.pkl
        where x and y are centre position of window on detector, 
        R is radius of defined PSF region, name is name of target used
        to derive PSF, visit its visit name, and lib_num a consecutive
        version number.
        """
        return self.library_name(lib_num, 
                            prefix='psfs',
                            postfix=f'{self.pps.name}_{self.pps.visit}_',
                            component=component)


    def eigen_name(self, lib_num):
        """Returns full path name to PSF eigenfunctions library. Name
        is of the format eigenlib_{x}_{y}_{R}_{lib_num}.pkl
        where x and y are centre position of window on detector, 
        R is radius of defined PSF region, and lib_num a consecutive
        version number.
        """
        return self.library_name(lib_num, 'eigenlib', component='')
    
    
    def library_name(self, lib_num, prefix, postfix='', component='A'):
        """Composes filename of PSF library. The file is assummed to 
        be located in the subdirectory 'psf_lib' to the reference path.
        """
        ixoff = self.sa_hdr['X_WINOFF'] + int(self.sa_debias[0].shape[0]/2)
        iyoff = self.sa_hdr['Y_WINOFF'] + int(self.sa_debias[0].shape[1]/2)
        return os.path.join(self.pps.calibpath,
                            'psf_lib',
                            '{:s}_{:d}_{:d}_{:d}_{:s}{:d}{:s}.pkl'.format(
                                prefix,
                                ixoff, iyoff, int(self.pps.psf_rad),
                                postfix, lib_num, component))
        
    
    def pre_process(self):
        """Pre-processing reduces the data (without extracting photometry).
        It subtracts the background, removes potential smearing, and also uses
        the PSF to compute more accurate centers.
        """
        self.mess('--- Start pre-processing')
        self.reduce_data_sa()
        self.sa_mask_cube = np.ones(self.sa_debias.shape, dtype='?')
        self.sa_mask_cube[:] = self.sa_mask
        
        # Define PSF normalised to subarray flux
        psf_flux = integrate_psf(self.psf, radius=self.pps.im_psfrad)
        sa_med_flux = self.median_flux_sa(radius=self.pps.im_psfrad)
        self.sa_norm = sa_med_flux/psf_flux
        self.sa_psf = lambda y, x, grid=True: self.sa_norm * self.psf(y, x, grid=grid)

        bg_psf_flux = integrate_psf(self.bg_psf, radius=self.pps.im_psfrad)
        self.sa_bg_norm = sa_med_flux/bg_psf_flux
        self.sa_bg_psf = lambda y, x, grid=True: self.sa_bg_norm * self.bg_psf(y, x, grid=grid)

        if self.pps.binary:
            skip_bg_stars = [0, self.pps.secondary]
        else:
            skip_bg_stars = [0]
        
        # Remove background stars and smearing, do a more precise re-centre
        # and remove background stars again for better fit
        self.make_star_bg_cube_sa(skip=skip_bg_stars)
        self.update_smearing_sa()
        self.update_sub_sa()
        self.psf_cent_sa()
        self.make_star_bg_cube_sa(skip=skip_bg_stars)
        self.update_sub_sa()

        # Reduce imagettes, if they exist
        if self.pps.file_im is not None:
            self.reduce_data_im()
            self.im_mask_cube = np.ones(self.im_raw.shape, dtype='?')
            self.im_mask_cube[:] = self.im_mask

            self.im_norm = sa_med_flux/(psf_flux * self.nexp)
            self.im_psf = lambda y, x, grid=True: self.im_norm * self.psf(y, x, grid=grid)
            self.im_bg_norm = sa_med_flux/(bg_psf_flux * self.nexp)
            self.im_bg_psf = lambda y, x, grid=True: self.im_bg_norm * self.bg_psf(y, x, grid=grid)

            self.make_star_bg_cube_im(skip=skip_bg_stars)
            self.update_smearing_im()
            self.update_sub_im()
            self.psf_cent_im()
            self.make_star_bg_cube_im(skip=skip_bg_stars)
            self.update_sub_im()


    def iter_background(self, res, bg):
        """In case a bright background star is near the target, an optimal
        factor for the background can be searched to better subtract the
        background star contribution.
        """
        from scipy.optimize import minimize_scalar
        self.mess('Iterating background')
        bgfact = np.ones(len(res))
        for n in range(len(res)):
            chi2 = lambda a : np.mean((res[n]+a*bg[n])**2)
            bgfact[n] = 1 - minimize_scalar(chi2, bracket=(-1.5,1.5))['x']
        return bgfact


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
        klip = self.pps.klip
        niter = self.pps.sigma_clip_niter
        if klip is None:
            klip = len(self.eigen_psf)
        else:
            klip = min(klip, len(self.eigen_psf))
        sel = self.sa_cent_sel
        
        # In case of very bright background stars close to the target,
        # a more careful fit of the background is warranted
        if (self.starcat.brightest(2*self.pps.sa_psfrad) > 0.1 and
            self.pps.bgstars):
            iter_bg = True
            niter += 1
        else:
            iter_bg = False

        bg = np.zeros_like(self.sa_bg)
        bg_fit = self.pps.bg_fit

        for n in range(niter):
            self.mess('--- Iteration sa {:d}/{:d}'.format(n+1, niter))
            psf_cube0, scale0, bg0, w0 = multi_psf_fit(
                            self.eigen_psf[:klip],
                            self.sa_sub[sel] - self.sa_stat_res - bg[sel, None, None],
                            self.sa_noise[sel],
                            self.sa_mask_cube[sel],
                            self.sa_xc[sel], self.sa_yc[sel],
                            fitrad=self.pps.fitrad,
                            defrad=self.pps.psf_rad,
                            bg_fit=bg_fit,
                            nthreads=self.pps.nthreads,
                            non_negative=self.pps.non_neg_lsq)
            # Interpolate over frames without source
            t0 = self.sa_att[sel, 0]
            t = self.sa_att[:, 0]        
            psf_cube = interp_cube_ext(t, t0, psf_cube0)
            w = interp_cube_ext(t, t0, w0)
            scale = np.interp(t, t0, scale0)
            bg += np.interp(t, t0, bg0)
            if (self.pps.bg_smo is not False) and (n == niter-1):
                bg[sel] = smo_spl_bg(self.sa_att[sel,0], bg[sel], smo_len=self.pps.bgfit_smo,
                                     smo_lim=self.pps.bgfit_smo_lim)
                bg_fit = -1
                
            self.mess('Iter {:d} MAD sa: {:.2f} ppm'.format(n+1, mad(scale0)))
            self.make_mask_cube_sa(psf_cube, bg)
            self.sa_mask_cube[sel==0] = self.sa_mask
            res = self.sa_sub - psf_cube - bg[:,None,None]
            self.compute_resid_stat_sa(res)
            res -= self.sa_stat_res
            self.remove_resid_smear_sa(res, psf_cube)

            if iter_bg:
                bgfact = self.iter_background(self.sa_sub - psf_cube,
                                              self.sa_bgstars)
                self.sa_bgstars *= np.nanmedian(bgfact)
                self.mess('Background factor: {:.3f} [sa]'.format(np.nanmedian(bgfact)))
                self.update_sub_sa()
                self.psf_cent_sa()
                
            self.sa_noise = (self.psf_noise_sa(psf_cube + self.sa_bg[:, None, None] +
                bg[:, None, None] + 2*self.sa_bgstars + np.abs(self.sa_smear[:, None, :]) + 
                np.abs(self.sa_stat_res) + np.abs(self.sa_dark))**2 +
                self.sa_dark_err**2)**.5

        flux, err = self.psf_phot_sa(psf_cube, bg, self.pps.im_psfrad-2)
        if self.pps.save_resid_cube:
            self.save_residuals_sa('', psf_cube, bg)

        if self.pps.save_bg_cube:
            self.save_bg_sa('', psf_cube, bg)            
        
        flagCenter = (self.filter_pos(self.sa_xc, self.sa_yc) == 0)
        flagBadPix = (self.filter_bad_masks(self.sa_mask_cube, self.sa_apt) == 0)
        flagFlux = (self.filter_flux(scale) == 0)
        flagSource = (sel == 0)
        flagBG = (self.filter_high_bg(bg + self.sa_bg, sel=(flagFlux==0)) == 0)
        flag = np.zeros(flagCenter.shape, dtype='int8')
        flag[:] = (1*flagCenter + 2*flagBadPix + 4*flagFlux +
                    8*flagSource + 16*flagBG)
        self.save_eigen_sa(flag, scale, scale*err,
                           bg + self.sa_bg, w)
        if self.pps.save_mask_cube:
            self.save_cube_fits('mask_cube_sa.fits', 
                                np.array(self.sa_mask_cube, dtype='uint8'))
        if self.pps.save_psfmodel:
            self.save_cube_fits('psf_model_sa.fits', psf_cube)

        if self.pps.save_static:
            self.save_cube_fits('static_sa.fits', self.sa_stat_res)

        if self.pps.save_noise_cubes:
            self.save_cube_fits('psf_noise_sa.fits', self.sa_noise)
            self.save_cube_fits('raw_noise_sa.fits', self.sa_raw_noise)
            res = self.compute_residuals_sa(psf_cube, bg)
            emp_noise_cube = empiric_noise(res, self.sa_xc, self.sa_yc, bg + self.sa_bg)
            self.save_cube_fits('empiric_noise_sa.fits', emp_noise_cube)

        sel = (flag==0)
        self.mad_sa = mad(scale)
        self.mess('MAD sa: {:.2f} ppm'.format(self.mad_sa))
        self.mess('MAD sa[flag==0]: {:.2f} ppm'.format(mad(scale[sel])))

        return  scale, bg, flux, err, sel, w


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
        klip = self.pps.klip
        niter = self.pps.sigma_clip_niter
        if klip is None:
            klip = len(self.eigen_psf)
        else:
            klip = min(klip, len(self.eigen_psf))
        sel = self.im_cent_sel
        
        # In case of very bright background stars close to the target,
        # a more careful fit of the background is warranted
        if (self.starcat.brightest(2*self.pps.im_psfrad) > 0.1 and
            self.pps.bgstars):
            iter_bg = True
            niter += 1
        else:
            iter_bg = False

        for n in range(niter):
            self.mess('--- Iteration im {:d}/{:d}'.format(n+1, niter))
            # Only extract photometry from frames with source
            psf_cube0, scale0, bg0, w0 = multi_psf_fit(
                            self.eigen_psf[:klip],
                            self.im_sub[sel] - self.im_stat_res,
                            self.im_noise[sel],
                            self.im_mask_cube[sel],
                            self.im_xc[sel], self.im_yc[sel],
                            fitrad=self.pps.fitrad,
                            defrad=self.pps.psf_rad,
                            bg_fit=self.pps.bg_fit,
                            nthreads=self.pps.nthreads, 
                            non_negative=self.pps.non_neg_lsq)
            psf_cube0 *= self.im_apt

            # Interpolate over frames without source
            t0 = self.im_att[sel, 0]
            t = self.im_att[:, 0]
            psf_cube = interp_cube_ext(t, t0, psf_cube0)
            w = interp_cube_ext(t, t0, w0)
            scale = np.interp(t, t0, scale0)
            bg = np.interp(t, t0, bg0)
            self.mess('Iter {:d} MAD im: {:.2f} ppm'.format(n+1, mad(scale0)))
            self.make_mask_cube_im(psf_cube, bg)
            self.im_mask_cube[sel==0] = self.im_mask
            self.compute_resid_stat_im(self.im_sub - psf_cube - bg[:,None,None])

            if iter_bg:
                bgfact = self.iter_background(self.im_sub - psf_cube, self.im_bgstars)
                self.im_bgstars *= np.nanmedian(bgfact)
                self.mess('Background factor: {:.3f} [im]'.format(np.nanmedian(bgfact)))
                self.update_sub_im()
                self.psf_cent_im()               
            self.im_noise = (self.psf_noise_im(psf_cube + self.im_bg[:, None, None] +
                bg[:, None, None] + 2*self.im_bgstars + self.im_smear[:, None, :] +
                np.abs(self.im_stat_res) + np.abs(self.im_dark))**2 +
                self.im_dark_err**2)**.5

        flux, err = self.psf_phot_im(psf_cube, bg, self.pps.im_psfrad-2)

        if self.pps.save_resid_cube:
            self.save_residuals_im('', psf_cube, bg)
        
        flagCenter = (self.filter_pos(self.im_xc, self.im_yc) == 0)
        flagBadPix = (self.filter_bad_masks(self.im_mask_cube, self.im_apt) == 0)
        flagFlux = (self.filter_flux(scale) == 0)
        flagBG = (self.filter_high_bg(bg + self.im_bg, sel=(flagFlux==0)) == 0)
        flagSource = (sel == 0)
        flag = np.zeros(flagCenter.shape, dtype='int8')
        flag[:] = (1*flagCenter + 2*flagBadPix + 4*flagFlux +
                    8*flagSource + 16*flagBG)
        self.save_eigen_im(flag, scale, scale*err, bg + self.im_bg, w)
        if self.pps.save_mask_cube:
            self.save_cube_fits('mask_cube_im.fits',
                                np.array(self.im_mask_cube, dtype='uint8'))
        if self.pps.save_psfmodel:
            self.save_cube_fits('psf_model_im.fits', psf_cube)

        if self.pps.save_static:
            self.save_cube_fits('static_im.fits', self.im_stat_res)

        if self.pps.save_noise_cubes:
            self.save_cube_fits('psf_noise_im.fits', self.im_noise)
            self.save_cube_fits('raw_noise_im.fits', self.im_raw_noise)
            res = self.compute_residuals_im(psf_cube, bg)
            emp_noise_cube = empiric_noise(res, self.im_xc, self.im_yc, bg + self.im_bg)
            self.save_cube_fits('empiric_noise_im.fits', emp_noise_cube)

        sel = (flag==0)
        self.mad_im = mad(scale)
        self.mess('MAD im: {:.2f} ppm'.format(self.mad_im))
        self.mess('MAD im[flag==0]: {:.2f} ppm'.format(mad(scale[sel])))
        
        return  scale, bg, flux, err, sel, w


    def reduce_data_sa(self, center=True):
        """Background subtracts the de-biased frames using a preliminary 
        background estimated from DRP products. The central 50% of each 
        framed is checked for a source, to filter out source-less frames.
        If center is True, a photocentre is computed for each frame with 
        a source.
        """
        self.mess('--- Starting reductions [sa]')
        if self.pps.non_lin_tweak:
            # This tweak should eventually be put into the regular 
            # non-linear correction curve
            self.mess('Tweaking non-linear correction for low exposures [sa]')
            def nonlin_tweak(f):
                ret = (f/self.sa_hdr['NEXP']-100)/(self.pps.nl_lim-100)*self.pps.nl_100 + 1 - self.pps.nl_100
                ret[f/self.sa_hdr['NEXP'] > self.pps.nl_lim] = 1
                return ret
            self.sa_debias /= nonlin_tweak(self.sa_debias)
            self.sa_bg /= nonlin_tweak(self.sa_bg)
            self.sa_dark /= nonlin_tweak(self.sa_dark)
        
        self.sa_sub = (self.sa_debias - self.sa_bg[:, None, None])*self.sa_apt
        if self.pps.darksub:
            self.sa_sub -= self.sa_dark
        
        self.sa_noise = self.raw_noise_sa()
        ind1 = int(0.25*self.sa_sub.shape[1])
        ind2 = int(0.25*self.sa_sub.shape[2])
        sel = self.filter_source(self.sa_sub[:,ind1:-ind1,ind2:-ind2],
                                 mask=self.sa_mask[ind1:-ind1,ind2:-ind2])
        self.sa_cent_sel = sel
        self.mess('No centering source: {:d} / {:d} [sa]'.format(np.sum(sel==0), len(sel)))
        if center:
            xc, yc = self.photo_cent(self.sa_sub[sel])
            # Interpolate centers for frames without source
            self.sa_xc = np.interp(self.sa_att[:, 0], self.sa_att[sel, 0], xc)
            self.sa_yc = np.interp(self.sa_att[:, 0], self.sa_att[sel, 0], yc)


    def reduce_data_im(self, center=True):
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
        self.sub_im_bias()
        self.apply_im_flatfield(self.pps.Teff)
        self.sa_bg2im_bg()
        self.im_sub = self.im_debias - self.im_bg[:, None, None]*self.im_apt
        if self.pps.darksub:
            self.im_sub -= self.im_dark
        
        self.im_noise = self.raw_noise_im()
        
        sel = self.filter_source(self.im_sub, mask=self.im_mask)
        self.im_cent_sel = sel
        self.mess('No centering source: {:d} / {:d} [im]'.format(np.sum(sel==0), len(sel)))


        if center:
            xc, yc = self.photo_cent(self.im_sub[sel])
            # Interpolate centers for frames without source
            self.im_xc = np.interp(self.im_att[:,0], self.im_att[sel,0], xc)
            self.im_yc = np.interp(self.im_att[:,0], self.im_att[sel,0], yc)

    def read_sa_datacube(self):
        """Reads the de-biased, flat-fielded, and non-linearity corrected
        subarray data cube from data file. Uses DRP products to estimate
        a prelininary background, corrected for a systematic background
        bias. Initilise the source coordinates to be at the centre of
        the frame.
        """
        self.mess('Reading de-biased data cube... [sa]')
        
        self.sa_debias, self.sa_mjd, self.sa_hdr, self.sa_tab = \
            datacube(self.pps.file_sa_cal, self.pps.sa_range)
        
        
        self.sa_apt = (np.max(self.sa_debias, axis=0) > 0)
        sa_cor, _sa_mjd, _sa_hdr, _sa_tab = \
            datacube(self.pps.file_sa_cor, self.pps.sa_range)

        self.sa_bg = np.median(self.sa_debias[:, self.sa_apt] - 
                               sa_cor[:, self.sa_apt], axis=1)
        # Correct for the bg offset from the DRP estimates
        inrad = 0.95*0.5*np.min(sa_cor[0].shape)
        apt95 = aperture(sa_cor[0].shape, radius=inrad)
        self.sa_ring95 = self.sa_apt * (apt95 == 0)
        bgoff = np.median(sa_cor[:,self.sa_ring95])
        self.sa_bg += bgoff
        self.sa_bgstars = np.zeros(self.sa_debias.shape)
        self.sa_xc = 0.5*self.sa_debias.shape[-2]*np.ones(self.sa_debias.shape[0])
        self.sa_yc = 0.5*self.sa_debias.shape[-1]*np.ones(self.sa_debias.shape[0])

        
    def read_imagettes(self, im_range=None):
        """Read imagette cube, but only frames in the specified range.
        If im_range=None, then all frames are read
        Initilise the source coordinates to be at the centre of the
        frame.
        """
        self.mess('Reading imagette data cube...')
        self.im_raw, self.im_mjd, self.im_hdr, self.im_tab = \
            datacube(self.pps.file_im, im_range)
        self.im_off, self.im_sa_off = imagette_offset(self.pps.file_im)
        self.im_apt = (np.max(self.im_raw, axis=0) > 0)
        self.im_bgstars = np.zeros(self.im_raw.shape)
        self.im_xc = 0.5*self.im_raw.shape[-2]
        self.im_yc = 0.5*self.im_raw.shape[-1]


    def read_mask(self):
        """Reads the DRP provided bad pixel map to determine what
        pixels to initially mask. Pixels outside of the field of view
        are always masked. Mask pixel values == 1 means the pixel is 
        not masked, while 0 means it will be masked out and not used.
        """
        self.sa_mask = self.sa_apt.copy()
        if self.pps.file_mask is not None:
            self.mess('Load mask...')
            mask = read_mask(self.pps.file_mask)
            if self.pps.mask_badpix:
                self.sa_mask *= (mask==0)            
            self.mess('{:d} pixels masked in subarray'.format(np.sum(self.sa_mask[self.sa_apt]==0)))
            if self.pps.file_im is not None:
                self.define_im_mask()


    def define_im_mask(self):
        """Crops the imagette part of the subarray mask to define the
        imagette mask.
        """
        self.im_mask = self.sa2im_crop(self.sa_mask) * self.im_apt
        self.mess('{:d} pixels masked in imagettes'.format(np.sum(self.im_mask[self.im_apt]==0)))


    def read_darks(self):
        """Read dark current estimates from directory with dark current files
        If no dark current is to be subtracted, define it to be zero.
        """
        if not self.pps.darksub:
            self.mess('Skipping dark frames...')
            self.sa_dark = np.zeros_like(self.sa_debias[0])
            self.sa_dark_err = self.sa_dark
            if self.pps.file_im is not None:
                self.im_dark = np.zeros_like(self.im_raw[0])
                self.im_dark_err = self.im_dark
            return
        self.mess('Reading dark frames...')
        dark, dark_err = read_dark(self.pps.calibpath,
                                   self.sa_hdr['V_STRT_M'],
                                   (self.sa_hdr['X_WINOFF'], self.sa_hdr['Y_WINOFF']),
                                   self.sa_debias.shape[1:])
        clean_mask = dark > self.pps.dark_level
        dark *= clean_mask
        dark_err *= clean_mask
        self.sa_dark = dark * self.sa_hdr['TEXPTIME'] * self.sa_apt
        self.sa_dark_err = dark_err * self.sa_hdr['TEXPTIME'] * self.sa_apt
        if self.pps.file_im is not None:
            i0, i1, j0, j1 = sub_image_indices(self.im_sa_off,
                                                    self.im_raw.shape[1:])
            self.im_dark = (dark[j0:j1, i0:i1] * self.im_hdr['NEXP'] *
                            self.im_hdr['EXPTIME'] * self.im_apt)
            self.im_dark_err = (dark_err[j0:j1, i0:i1] * self.im_hdr['NEXP'] *
                                self.im_hdr['EXPTIME'] * self.im_apt)


    def read_attitude(self):
        """Reads attitude data from attitude file and resamples it
        to imagette cadence if imagettes are defined
        """
        self.mess('Reading attitude data...')
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


    def read_gain(self):
        """Defines the gain in e/ADU for each imagette to either a
        given value or computed from HK.
        """
        if self.pps.gain is None:
            self.mess('Reading HK and computing gain...')
            self.pps.gain = np.median(read_gain(self.pps.file_hk, self.pps.file_gain))

            #self.mess('Gain = {:.4f} e/ADU +/- {:.1f} ppm'.format(self.pps.gain,
            #            1e6*np.std(self.im_gain)/self.pps.gain))
        if self.pps.file_im is not None:
            self.im_gain = self.pps.gain * np.ones(len(self.im_raw))
            self.mess('Defined gain = {:.4f} e/ADU'.format(self.pps.gain))


    def read_ron(self):
        """Reads read-out noise from DRP-calibrated subarray file,
        returns RON in electrons. Gain must first be defined.
        """
        if self.pps.ron is None:
            self.mess('Reading subarray read-out noise...')
            if self.pps.gain is None:
                raise Exception('No gain defined')
            ron = read_ron(self.pps.file_sa_cal)
            self.pps.ron = ron*self.pps.gain
        if self.pps.ron > 0:
            self.mess('RON = {:.2f} ADU'.format(self.pps.ron))
        else:
            self.mess(f'WARNING: RON = {self.pps.ron} ADU')


    def read_thermFront_2(self):
        """Reads the thermFront_2 sensor data from the raw subarray
        data cube.
        """
        self.mess('Reading thermFront_2 sensor data...')
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
            self.mess('Skipping star catalogue...')
            return
        self.mess('Loading star catalogue...')
        shape = self.sa_debias[0].shape
        maxrad = (np.max(shape)**2 + np.max(self.pps.ccdsize)**2)**0.5
        self.starcat = star_bg(self.pps.file_starcat,
                                        maxrad=maxrad,
                                        shape=shape)


    def define_mjd2bjd(self):
        """ Use table from DRP-produced file to produce MJD->BJD
        function.
        """
        self.mess('Defining MJD to BJD conversion...')
        if self.pps.mjd2bjd:
            self.mjd2bjd = mjd2bjd(self.pps.file_sa_cal)
        else: # Don't use barycentric conversion
            self.mjd2bjd = lambda mjd : mjd + 2400000.5


    def sub_im_bias(self):
        """Uses bias values listed in subarray table to interpolate
        the biases on the imagette cadence.
        """
        if self.pps.bias is None:
            bias = self.im_hdr['NEXP'] * interp_cube(self.im_mjd, self.sa_mjd,
                                                            self.sa_tab['BIAS'])
        else:
            bias = np.ones_like(self.im_mjd)*self.pps.bias
        self.im_debias = self.adu2e_im((self.im_raw - bias[:,None,None])
                                       *self.im_apt)
        
        
    def adu2e_im(self, im_adu):
        """Applies a non-linearity function and the gain to convert from
        detected ADU to number of electrons.
        """
        self.mess('Converting ADU to electrons [im]...')
        if self.pps.non_lin:
            self.mess(' Using non-linearity correction [im]...')
            nonlinfun = nonlinear(self.pps.file_nonlin)
            im_e = im_adu * nonlinfun(im_adu/self.im_hdr['NEXP']) * self.im_gain[:,None,None]
        else:
            self.mess(' No non-linearity correction [im]...', level=2)
            im_e = im_adu  * self.im_gain[:,None,None]            
        if self.pps.non_lin_tweak:
            # This tweak should eventually be put into the regular 
            # non-linear correction curve
            self.mess('Tweaking non-linear correction for low exposures [sa]')
            def nonlin_tweak(f):
                ret = (f/self.im_hdr['NEXP']-100)/(self.pps.nl_lim-100)*self.pps.nl_100 + 1 - self.pps.nl_100
                ret[f/self.im_hdr['NEXP'] > self.pps.nl_lim] = 1
                return ret
            im_e /= nonlin_tweak(im_e)
        return im_e


    def apply_im_flatfield(self, target_Teff):
        """Use the flatfield reference file to correct the
        imagette data for flatfield
        """
        if self.pps.flatfield:
            self.mess('Reading and applying flatfield [im]...')
            iff = flatfield(self.pps.file_flats, target_Teff,
                                 self.im_off, self.im_debias[0].shape)
            self.im_debias /= iff
        else:
            self.mess('No flatfield applied [im].', level=2)

        
    def update_smearing_im(self):
        """Compute the smearing correction using stars from the
        star catalogue, and the given imagette PSF.
        """
        self.im_smear = np.zeros(self.im_raw.shape[0:2])
        if not self.pps.smear_corr:
            self.mess('No smearing correction update [im].', level=2)
            return
        self.mess('Estimate smearing [im]...')
        yoff = self.im_off[1]
        shape = (self.pps.ccdsize[0], self.im_sub.shape[2])
        for n in range(len(self.im_sub)):
            xc = self.im_xc[n]
            yc = yoff + self.im_yc[n]
            self.im_smear[n] = self.pps.smear_fact * self.starcat.smear(xc, yc,
                                                          self.im_att[n,3],
                                                          self.im_att[n,4],
                                                          self.im_psf,
                                                          shape=shape)
        self.im_smear /= (self.im_hdr['EXPTIME'] * self.sa_hdr['RO_FREQU'])


    def update_smearing_sa(self):
        """Compute the smearing correction using stars from the
        star catalogue, and the given subarray PSF.
        """
        self.sa_smear = np.zeros(self.sa_debias.shape[0:2])
        if not self.pps.smear_corr:
            self.mess('No smearing correction update [sa].', level=2)
            return
        self.mess('Estimate smearing [sa]...')
        yoff = self.sa_hdr['Y_WINOFF']
        shape = (self.pps.ccdsize[0], self.sa_sub.shape[2])
        for n in range(len(self.sa_sub)):
            xc = self.sa_xc[n]
            yc = yoff + self.sa_yc[n]
            self.sa_smear[n] = self.pps.smear_fact * self.starcat.smear(xc, yc,
                                                          self.sa_att[n,3],
                                                          self.sa_att[n,4],
                                                          self.sa_psf,
                                                          shape=shape)
        self.sa_smear /= (self.sa_hdr['EXPTIME']*self.sa_hdr['RO_FREQU'])


    def compute_resid_stat_sa(self, res):
        """Compute the static part of the residuals
        """
        if self.pps.remove_static:
            nanres = res.copy()
            if self.pps.static_psf_rad:
                apts_in = cube_apt(res.shape, self.pps.fitrad, self.sa_xc, self.sa_yc)
                nanres[apts_in] = np.nan
                apts_out = cube_apt(res.shape, self.pps.psf_rad - 1, self.sa_xc, self.sa_yc)
                nanres[apts_out==0] = np.nan
            else:
                apts = self.sa_apt
                nanres[:,apts==0] = np.nan
            nanres[0,:,:] = 0   # Ensures not all values are nan
            self.sa_stat_res = np.nanmedian(nanres, axis=0)
            self.sa_stat_res *= (self.sa_stat_res > 0)


    def compute_resid_stat_im(self, nanres):
        """Compute the static part of the residuals
        """

        if self.pps.remove_static:
            if self.pps.static_psf_rad:
                return
            apts = self.im_apt
            nanres[:, apts==0] = np.nan
            nanres[0,:,:] = 0   # Ensures not all values are nan
            self.im_stat_res = np.nanmedian(nanres, axis=0)
            self.im_stat_res *= (self.im_stat_res > 0)


    def remove_resid_smear_sa(self, res, psf_cube, spotfrac=1e-4):
        """Looks at resduals along columns to remove any residual 
        smearing. Masks brightest region of PSF in estimate.
        """
        if not self.pps.smear_resid:
            self.mess('Not removing residual smear [sa].', level=2)
            return
        self.mess('Removing residual smear... [sa]')
        spot = (psf_cube/self.sa_norm) > spotfrac
        ns = spot.shape[1]*spot.shape[2]
        mss = np.max(np.sum(spot,axis=(1,2)))
        self.mess(' Spot fraction: {:d}/{:d} = {:.3f}%'.format(mss,
                                                             ns,
                                                             100*mss/ns))
        apt = aperture(self.sa_sub[0].shape, self.pps.psf_rad)
        nanres = res.copy()
        nanres[:, apt==0] = np.nan
        nanres[spot] = np.nan
        nanres[:,int(nanres.shape[1]/2),:] = 0
        self.sa_smear += np.nanmedian(nanres, axis=1)
        self.update_sub_sa()


    def remove_resid_smear_im(self, res, psf_cube, spotfrac=1e-4):
        """Looks at resduals along columns to remove any residual 
        smearing. Masks brightest region of PSF in estimate.
        Warning: this can result in a noisy correction for imagettes
        where the field outside PSF is small.
        """
        if not self.pps.smear_resid:
            self.mess('Not removing residual smear [im].', level=2)
            return
        self.mess('Removing residual smear... [im]')
        spot = (psf_cube/self.im_norm) > spotfrac
        nanres = res.copy()
        nanres[:, self.im_apt==0] = np.nan
        ns = spot.shape[1]*spot.shape[2]
        mss = np.max(np.sum(spot,axis=(1,2)))
        self.mess(' Spot fraction: {:d}/{:d} = {:.3f}%'.format(mss,
                                                             ns,
                                                             100*mss/ns))
        nanres[spot] = np.nan
        nanres[:,int(nanres.shape[1]/2),:] = 0
        self.im_smear += np.nanmedian(nanres, axis=1)
        self.update_sub_im()


    def sa_bg2im_bg(self):
        """Define an initial background estimate for the imagettes by
        interpolating the already known background for the subarrays.
        """
        self.mess('Interpolate background from subarray to imagettes...')
        self.im_bg = np.interp(self.im_mjd, self.sa_mjd, self.sa_bg) / self.nexp


    def update_sub_sa(self):
        """Produce a new background-subtracted image
        """
        self.mess('Subtracting bg stars from sub cube...')        
        self.sa_sub = (self.sa_debias - self.sa_bg[:, None, None])
        if self.pps.bgstars:
            self.sa_sub -= self.sa_bgstars
        if self.pps.smear_corr:
            self.sa_sub -= self.sa_smear[:, None, :]
        if self.pps.drp_bg_corr:
            # DRP shows a systemativ oversubtraction of background, this
            # aims to correct for that by computing a constant to offset
            # the median of the background-subtracted images to zero
            # at R = 95 pix
            bgoff = np.median(self.sa_sub[:,self.sa_ring95])
            self.sa_bg += bgoff
            self.sa_sub -= bgoff
        self.sa_sub *= self.sa_apt

    
    def update_sub_im(self):
        """Produce a new background-subtracted image
        """
        self.mess('Subtract background... [im]')
        self.im_sub = self.im_debias - self.im_bg[:,None,None]
        if self.pps.bgstars:
            self.im_sub -= self.im_bgstars
        if self.pps.smear_corr:
            self.im_sub -= self.im_smear[:, None, :]
        # Add signal outside aperture, to simplify 2D-splines fits over square
        self.im_sub *= self.im_apt
        self.im_sub += (self.im_apt==0)*np.min(
                np.median(self.im_sub, axis=0)[self.im_apt])

        
    def raw_noise_sa(self):
        """Compute noise from debiased data, assuming poisson noise, 
        read-out noise, and dark current noise (when applicable)
        """
        self.mess('Compute noise cube [sa]...')
        iron = self.pps.ron*self.sa_hdr['NEXP']**0.5
        self.sa_raw_noise = noise(self.sa_debias, ron_elec=iron)
        if self.pps.darksub:
            self.sa_raw_noise = (self.sa_raw_noise**2 + self.sa_dark_err**2)**.5
        # Define noise outside aperture
        self.sa_raw_noise *= self.sa_apt
        self.sa_raw_noise += (self.sa_apt==0)*np.max(np.median(self.sa_raw_noise, axis=0))
        return self.sa_raw_noise


    def raw_noise_im(self):
        """Compute noise from debiased data, assuming poisson noise, 
        read-out noise, and dark current noise (when applicable)
        """
        self.mess('Compute noise cube [im]...')
        iron = self.pps.ron*self.im_hdr['NEXP']**0.5
        self.im_raw_noise = noise(self.im_debias, ron_elec=iron)
        if self.pps.darksub:
            self.im_raw_noise = (self.im_raw_noise**2 + self.im_dark_err**2)**.5
        # Define noise outside aperture
        self.im_raw_noise *= self.im_apt
        self.im_raw_noise += (self.im_apt==0)*np.max(np.median(self.im_raw_noise, axis=0))
        return self.im_raw_noise


    def psf_noise_sa(self, src_mod_cube):
        """Compute noise assuming a model distribution of flux
        """
        self.mess('Compute PSF noise [sa]...')
        iron = self.pps.ron*self.sa_hdr['NEXP']**0.5
        self.sa_mod_noise = psf_noise(src_mod_cube, ron_elec=iron)
        # Define noise outside aperture
        self.sa_mod_noise *= self.sa_apt
        self.sa_mod_noise += (self.sa_apt==0)*np.max(np.median(self.sa_mod_noise, axis=0))
        return self.sa_mod_noise


    def psf_noise_im(self,  src_mod_cube):
        """Compute noise assuming a model distribution of flux
        """
        self.mess('Compute PSF noise [im]...')
        iron = self.pps.ron*self.im_hdr['NEXP']**0.5
        self.im_mod_noise = psf_noise(src_mod_cube, ron_elec=iron)
        # Define noise outside aperture
        self.im_mod_noise *= self.im_apt
        self.im_mod_noise += (self.im_apt==0)*np.max(np.median(self.im_mod_noise, axis=0))
        return self.im_mod_noise

        
    def photo_cent(self, data, radius=None):
        """Find photo centre for each frame of a cube, using an aperture
        centered in the frame (with defined radius)
        """
        if radius == None:
            radius = self.pps.fitrad
        self.mess('Compute photo centers...')
        apt = aperture(data[0].shape, radius=radius)
        ixc, iyc = cent_flux(data*apt)
        self.mess(' Cent std: ({:.3f}, {:.3f})'.format(np.std(ixc),
                                                  np.std(iyc)))
        return ixc, iyc


    def psf_cent_sa(self):
        """Use PSF fitting to find centres in subarrays. Interpolate positions
        over frames without source.
        """
        sel = self.sa_cent_sel        
        self.mess('Compute PSF centers [sa]... (multi {:d} threads)'.format(self.pps.nthreads))
        xc, yc = multi_cent_psf(self.psf,
                                self.sa_sub[sel] - self.sa_stat_res,
                                self.sa_noise[sel],
                                self.sa_xc[sel], self.sa_yc[sel],
                                mask=self.sa_mask, radius=self.pps.fitrad,
                                norm=self.sa_norm,
                                nthreads=self.pps.nthreads)
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
        self.mess('Compute PSF centers [im]... (multi {:d} threads)'.format(self.pps.nthreads))
        xc, yc = multi_cent_psf(self.psf,
                                self.im_sub[sel] - self.im_stat_res, 
                                self.im_noise[sel],
                                self.im_xc[sel], self.im_yc[sel],
                                mask=self.im_mask, radius=self.pps.fitrad,
                                norm=self.im_norm,
                                nthreads=self.pps.nthreads)
        self.im_xc = np.interp(self.im_att[:,0], self.im_att[sel, 0], xc)
        self.im_yc = np.interp(self.im_att[:,0], self.im_att[sel, 0], yc)
        ind = sigma_clip(xc) * sigma_clip(yc)
        self.mess('IM cent std: ({:.3f}, {:.3f})'.format(np.std(xc[ind]),
                                                     np.std(yc[ind])))       


    def make_pixtab_sa(self, sel, mask, fscale=None):
        """Produce a table of coordinates relative PSF centre, flux, and flux
        error per pixel. Used to derive a PSF.
        """
        self.mess('Produce pixel table [sa]...')
        return pix_mat(self.sa_sub[sel] - self.sa_stat_res, self.sa_noise[sel],
                              self.sa_xc[sel], self.sa_yc[sel], mask, fscale)

    
    def make_pixtab_im(self, sel, mask, fscale=None):
        """Produce a table of coordinates relative PSF centre, flux, and flux
        error per pixel. fscale is a normalisation array for the flux and the
        error, to put them on equal scale. Used to derive a PSF.
        """
        self.mess('Produce pixel table [im]...')
        return pix_mat(self.im_sub[sel] - self.im_stat_res, self.im_noise[sel],
                              self.im_xc[sel], self.im_yc[sel], mask, fscale)


    def median_flux_sa(self, radius=23):
        """Computes the median flux of the source within the
        aperture radius. Used to scale PSF of expected
        background sources.
        """
        flux = self.apt_phot_sa(radius=radius)
        sel = self.filter_flux(flux)
        sel *= self.filter_pos(self.sa_xc, self.sa_yc)
        return np.median(flux[sel])


    def median_flux_im(self, radius=23):
        """Computes the median flux of the source within the
        aperture radius. Used to scale PSF of expected
        background sources.
        """
        flux = self.apt_phot_im(radius=radius)
        sel = self.filter_flux(flux)
        sel *= self.filter_pos(self.im_xc, self.im_yc)
        return np.median(flux[sel])
    
    
    def apt_phot_sa(self, radius=23):
        """Computes rough aperture photometry (i.e. no edge interpolation, 
        simple interpolation of bad pixels) of radius on each subarray. 
        Returns array of flux.
        """
        clean = clean_cube2D(self.sa_sub, self.sa_mask, self.sa_apt)
        xc0 = np.median(self.sa_xc)
        yc0 = np.median(self.sa_yc)
        dx, dy = (xc0 - self.sa_xc), (yc0 - self.sa_yc)
        apt = aperture(clean[0].shape, radius=radius, xc=xc0, yc=yc0)
        flux = np.zeros(len(clean))
        for n in range(len(clean)):
            flux[n] = np.sum(shift(clean[n], (dy[n], dx[n]), order=1)[apt])
        return flux

    def apt_phot_im(self, radius=23):
        """Computes rough aperture photometry (i.e. no edge interpolation, 
        simple interpolation of bad pixels) of radius on each imagette. 
        Returns array of flux.
        """
        clean = clean_cube2D(self.im_sub, self.im_mask, self.im_apt)
        xc0 = np.median(self.im_xc)
        yc0 = np.median(self.im_yc)
        dx, dy = (xc0 - self.im_xc), (yc0 - self.im_yc)
        apt = aperture(clean[0].shape, radius=radius, xc=xc0, yc=yc0)
        flux = np.zeros(len(clean))
        for n in range(len(clean)):
            flux[n] = np.sum(shift(clean[n], (dy[n], dx[n]), order=1)[apt])
        return flux

        
    def psf_phot_sa(self, psf_cube, bg, radius):
        """Uses pre-defined PSFs to compute a noise-weighted sum of the flux
        of each subarray. Not as accurate as PSF fitting, but provides a
        good estimate of the expected error. Returns arrays of flux and
        its error estimate.
        """
        apt = self.apt_cube_sa(radius) * self.sa_mask_cube
        sa_noise = np.maximum(self.sa_raw_noise, self.sa_mod_noise)
        f, e = psf_phot_cube(self.sa_sub, sa_noise, psf_cube, bg, apt)
        return f, e


    def psf_phot_im(self, psf_cube, bg, radius):
        """Uses pre-defined PSFs to compute a noise-weighted sum of the flux
        of each imagette. Not as accurate as PSF fitting, but provides a
        good estimate of the expected error. Returns arrays of flux and
        its error estimate.
        """
        apt = self.apt_cube_im(radius) * self.im_mask_cube
        im_noise = np.maximum(self.im_raw_noise, self.im_mod_noise)
        f, e = psf_phot_cube(self.im_sub, im_noise, psf_cube, bg, apt)
        return f, e
        
    
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
        apt = np.zeros(self.im_raw.shape, dtype='?')
        shape = apt[0].shape
        for n in range(len(apt)):
            apt[n] = aperture(shape, radius, self.im_xc[n], self.im_yc[n])
        return apt


    def make_mask_cube_sa(self, psf_cube, bg):
        """Use a model of how the data should look like, the expected
        noise, and sigma-clipping to mask too deviating pixels 
        (e.g. cosmic rays). Only look for bad pixels inside radius.
        """
        if not self.pps.mask_badpix:
            self.mess('No mask cube [im].', level=2)
            return
        self.mess('Make mask cube [sa]...')
        if self.pps.empiric_noise:
            self.mess('Using empiric noise for mask cube [sa]...')
            res = self.compute_residuals_sa(psf_cube, bg)
            noise_cube = empiric_noise(res, self.sa_xc, self.sa_yc, bg + self.sa_bg)
            clip = self.pps.empiric_sigma_clip
        else:
            noise_cube = self.sa_noise
            clip = self.pps.sigma_clip

        self.sa_mask_cube = make_maskcube(self.sa_sub, noise_cube, psf_cube + bg[:,None,None],
                                                 mask=self.sa_mask, clip=clip)


    def make_mask_cube_im(self, psf_cube, bg):
        """Use a model of how the data should look like, the expected
        noise, and sigma-clipping to mask too deviating pixels 
        (e.g. cosmic rays). Only look for bad pixels inside radius.
        """
        if not self.pps.mask_badpix:
            self.mess('No mask cube [im].', level=2)
            return
        self.mess('Make mask cube [im]...')
        if self.pps.empiric_noise:
            self.mess('Using empiric noise for mask cube [im]...')
            res = self.compute_residuals_im(psf_cube, bg)
            noise_cube = empiric_noise(res, self.im_xc, self.im_yc, bg + self.im_bg)
            clip = self.pps.empiric_sigma_clip
        else:
            noise_cube = self.im_noise
            clip = self.pps.sigma_clip
        self.im_mask_cube = make_maskcube(self.im_sub, noise_cube, psf_cube + bg[:,None,None],
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
        self.mess('Compute star background cube [sa]...')
        for n in range(len(self.sa_sub)):
            self.sa_bgstars[n] = self.starcat.rotblur(self.sa_xc[n],
                                           self.sa_yc[n],
                                           rolldeg=self.sa_att[n,3],
                                           blurdeg=self.sa_att[n,4],
                                           psf_fun=self.sa_bg_psf,
                                           shape=self.sa_debias[0].shape,
                                           skip=skip,
                                           limflux=self.pps.limflux) * self.sa_apt


    def make_star_bg_cube_im(self, skip=[0]):
        """Produces image cube of properly located background stars, to
        be subtracted from observations. skip is a list of stars to be skipped;
        typically the target itself (0), but also the secondary in case of
        a binary.
        """
        if not self.pps.bgstars:
            self.mess('No background stars computed [im].', level=2)
            return
        self.mess('Compute star background cube [im]...')
        for n in range(len(self.im_sub)):
            self.im_bgstars[n] = self.starcat.rotblur(self.im_xc[n],
                                           self.im_yc[n],
                                           rolldeg=self.im_att[n,3],
                                           blurdeg=self.im_att[n,4],
                                           psf_fun=self.im_bg_psf,
                                           shape=self.im_debias[0].shape,
                                           skip=skip,
                                           limflux=self.pps.limflux) * self.im_apt


    def filter_source(self, data, mask=None, clip=5, niter=10):
        """Find out what frames are missing a source, and return
        a binary index array to indicate frames with source.
        """
        self.mess('Finding frames with source missing...')
        contrast = rough_contrast(data, mask=mask)
        sel, ret_str =  check_low(contrast, clip, niter)
        self.mess(ret_str, 2)
        return sel


    def filter_flux(self, flux, clip=5, niter=10):
        """Verifies the consistency of the photometry and de-selects
        data points with too deviating flux. Returns a binary index array
        of consistent data points.
        """
        self.mess('Filtering out frames with too deviating summed flux...')
        sel, ret_str = check_val(flux, clip, niter)
        self.mess(ret_str, 2)
        return sel

    
    def filter_pos(self, xc, yc, clip=5, niter=3):
        """Returns a binary index to positions without
        the outliers.
        """
        self.mess('Finding frames with excessive offsets...')
        sel, ret_str = check_pos(xc, yc, clip, niter)
        self.mess(ret_str, 2)
        return sel
    
    
    def filter_motion(self, xc, yc, lowfrac=0.9):
        """Selects frames with positions after and before that are fairly 
        consistent, with the goal to select frames with little 
        motion blur. lowfrac is the fraction of frames to keep,
        per coordinate.
        """
        self.mess('Filtering out frames with suspected motion...')
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
        self.mess('Finding frames with excessively masked pixels...')
        sel, ret_str = check_mask(mask_cube, apt, clip, niter)
        self.mess(ret_str, 2)
        return sel
    
        
    def filter_high_bg(self, bg, clip=10, niter=10, sel=None):
        """Returns binary index of frames without too deviating background,
        as often happens when pointing is close to Earth limb
        """
        self.mess('Finding frames with deviating background...')
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
        i1 = int(i0+self.im_raw[0].shape[1])
        j0 = int(self.im_sa_off[1])
        j1 = int(j0+self.im_raw[0].shape[0])

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
        self.mess('Reading T_EFF from starcat file ...')
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
        hdu = fits.PrimaryHDU(cube)
        hdul = fits.HDUList([hdu])
        hdul.writeto(os.path.join(self.pps.outdir, filename), overwrite=True)
        
        
    def save_psf(self, filename, psf_fun, name=''):
        self.mess('Saving {:s} PSF to \'{:s}\''.format(name, filename))
        pickle.dump(psf_fun, open(filename, 'wb'))        


    def compute_residuals_sa(self, psf_cube, bg):
        """Subtract model from data and return cube
        of residuals
        """
        model = psf_cube + bg[:,None,None]
        res = self.sa_sub - model
        res -= self.sa_stat_res
        return res


    def compute_residuals_im(self, psf_cube, bg):
        """Subtract model from data and return cube
        of residuals
        """
        model = psf_cube + bg[:,None,None]
        res = self.im_sub - model
        res -= self.im_stat_res
        return res


    def save_residuals_sa(self, prefix, psf_cube, bg):
        """Subtract model including bg stars from data and 
        save as a fits cube in the output directory.
        """
        res = self.compute_residuals_sa(psf_cube=psf_cube, bg=bg)
        self.save_cube_fits(prefix+'residuals_sa.fits', res)

    def save_bg_sa(self, prefix, psf_cube, bg):
        """Subtract model (but not bg stars) from data and save 
        as a fits cube in the output directory.
        """
        res = self.compute_residuals_sa(psf_cube=psf_cube, bg=bg)
        res += self.sa_bgstars
        self.save_cube_fits(prefix+'bgresiduals_sa.fits', res)


    def save_residuals_im(self, prefix, psf_cube, bg):
        """Subtract model from data and save as a fits cube
        in the output directory.
        """
        res = self.compute_residuals_im(psf_cube=psf_cube, bg=bg)
        self.save_cube_fits(prefix+'residuals_im.fits', res)    

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


    def pre_binary(self, integ_rad=15):
        """Special pre-processing for a binary. integ_rad is the integration 
        radius to be used for normalisation of PSFs and depends a bit on the
        separation. Can be tricky for strongly unresolved binaries, but is
        only an initial guess anyway so not critical.
        """
        self.pre_process()
        psf_flux0 = integrate_psf(self.psf0, radius=integ_rad)
        psf_flux1 = integrate_psf(self.psf1, radius=integ_rad)
        sa_med_flux = self.median_flux_sa(radius=integ_rad)
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
        
        self.centre_binary(self.psf)
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
            self.remove_resid_smear_sa(res, (psf_cube0+psf_cube1))
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
        
        self.im_xc0, self.im_yc0 = self.im_xc, self.im_yc
        dx, dy = rotate_position(self.binary_x1, self.binary_y1, self.im_att[:,3])
        self.im_xc1, self.im_yc1 = self.im_xc0 + dx, self.im_yc0 + dy
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


    def centre_binary(self, psf):
        dx, dy = self.starcat.rotate_entry(self.pps.secondary, self.sa_att[:,3])
        xc1, yc1 = self.sa_xc + dx, self.sa_yc + dy
        norm0 = self.sa_norm0
        norm1 = self.pps.init_flux_ratio*self.sa_norm1
        
        self.mess('Compute subarray PSF centers... (multi {:d} threads)'.format(self.pps.nthreads))
        sa_xc0, sa_yc0, _sc0, sa_xc1, sa_yc1, _sc1 = (
                multi_cent_binary_psf(psf,
                                      self.sa_sub, self.sa_noise,
                                      self.sa_xc, self.sa_yc,
                                      xc1, yc1,
                                      norm0, norm1,
                                      self.sa_mask_cube,
                                      nthreads=self.pps.nthreads))        
            
        x, y = derotate_position(sa_xc1-sa_xc0, sa_yc1-sa_yc0, self.sa_att[:,3])
        self.binary_x1, self.binary_y1 = np.median(x), np.median(y)
        ds = (x**2+y**2)**.5

        self.separation = np.median(ds)
        print('Astrometry: separation = {:.3f} +/- {:.3f} pix'.format(
                self.separation, np.std(ds)/len(ds)**.5))
        
        dx, dy = rotate_position(self.binary_x1, self.binary_y1, self.sa_att[:,3])
        
        self.mess('Compute subarray PSF centers... (multi {:d} threads)'.format(self.pps.nthreads))
        self.sa_xc0, self.sa_yc0, _sc0, self.sa_xc1, self.sa_yc1, _sc1 = (
                multi_cent_binary_psf_fix(psf,
                                          self.sa_sub, self.sa_noise,
                                          self.sa_xc, self.sa_yc, 
                                          dx, dy,
                                          norm0, norm1, 
                                          self.sa_mask_cube,
                                          nthreads=self.pps.nthreads))        


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
    
