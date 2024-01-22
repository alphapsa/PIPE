# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:55:41 2021

@author: Alexis Brandeker, alexis@astro.su.se

All input parameters are collected in the PipeParam class with
default values. Some configuration (for data filepaths) is also
present here, but should be moved into a configuration file.
"""
import os


class PipeParam:
    """ PipeParam keeps all parameters relevant for the extraction, with
    default values given for most.
    """
    def __init__(self, name, visit, version=None, outdir=None, sa_range=None,
                 datapath=None, calibpath=None):
        """ Initializes parameters with their default values. Name is the
        subdirectory to datapath where the data for the present visit resideds.
        Version is the version of the output files. If left to the default None,
        it will look for the highest existing version and add 1.
        """
        from .config import conf
        self.name = name         # Name of target; also name of
                                 # subdirectory with visits
        self.visit = visit       # Name of visit; also name of
                                 # subdirectory with data
        if datapath is None:
            self.datapath = os.path.join(conf.data_root, name, visit) # Directory path of data
                                                            # from single visit
        else:
            self.datapath = datapath
        if version is None:      # Integer version of output files
            self.version = self.find_version()
        else:
            self.version = version
        if outdir is None:       # Directory where output files are put
            outdir = os.path.join(self.datapath, 'Outdata', '{:05d}'.format(self.version))
        self.outdir = outdir
        
        self.file_log = os.path.join(outdir, 'logfile.txt')
        if calibpath is None:
            self.calibpath = conf.ref_lib_data  # Directory where calibration files are located
        else:
            self.calibpath = calibpath
        self.define_filenames()  # Find relevant filenames in data paths
        
        self.plog_verbosity = 1  # Decides level of verbosity for printed messages
        self.Teff = None         # Effective temperature of target, used for 
                                 # finding appropriate flat field.
        self.gain = None         # gain in electrons/ADU; estimated if not defined
        self.gain_median = True  # Use the median gain of visit for all frames
        self.ron = None          # read-out noise in RMS electrons per
                                 # readout and pixel; estimated if not defined
        self.bias = None         # bias in ADU; estimated if not defined
        self.sa_range = sa_range # tuple of 2 integers: Range of subarray indices to
                                 # be considered. If "None", the full array is used.
        self.mjd2bjd = True      # Use barycentric conversion from MJD to BJD [BUG in astropy 5.1 to be circumvented]
        self.binary = False      # If the binary code branch is to be used, requires
                                 # special parameters to be defined, see below
        self.psf_score = None    # Limits how good the PSF match needs to be
                                 # (lower score = stricter match)
        self.psf_min_num = 10    # Minimum number of PSFs used, irrespective of score
        self.psf_rad = 200       # PSF lib defined out to this integer radius
        self.psf_filenames_file = None  # If defined, the PSF will be defined from this
                                        # text file with PSF library files
        self.nthreads = os.cpu_count()-1    # Number of threads to use; defaulted to 
                                            # the number of system virtual cores - 1
        self.nthreads_blas = 1  # On some systems, BLAS is multithreaded. This is the number of
                                # estimated threads it uses. Used to not swamp thread pool.
                                # On windows, BLAS seems to use 6 threads, on linux 1 thread.
        self.bg_fit = 0          # Simultaneous background to be fit with PSF:
                                 # -1: no, 0: constant, [TBD: 1: bilinear plane, 2: parabolic]
        self.resample_im_times = False   # True if the time stamps for imagettes should
                                         # be corrected (early SOC bug) by interpolating
                                         # subarray times stamps. Not relevant for data
                                         # reprocessed 2021 or later.
        self.bgstars = True      # True if background stars are to be subtracted
        self.limflux = 1e-5      # Include bgstars down to this fraction of the target flux
        self.star_rad_scale = 1.0 # Use this multiplicative factor in heuristic for BG PSF radii
        self.fit_bgstars = True  # True if background stars are to be fitted
        self.lim_fit = 0.01      # Attempt to fit for position and photometry for bgstars down
                                 # to this fraction of target flux                                 
        self.bg_star_inrad = 5   # Minimum distance from targets for bg stars to be fitted
        self.blur_res = 0.5      # For bg stars, produce rotational blur to this pixel resolution
        self.mask_bg_stars = False       # Mask out pixels contaminated by background stars
        self.mask_bg_star_sep = 30       # min separation to target [in pix] for BG star being masked
        self.mask_bg_stars_circle = True # Use a circular mask, otherwise PSF-shaped
        self.mask_bg_radius = 20         # Mask out BG stars to this radius in pixels (if masked out)
        self.mask_bg_level = 0.1         # If not circular mask, use this level of peak to define mask
        self.centre = True       # If True, find the centre of the target star. Else, assume it is 
                                 # at the centre of the frame (plus default offsets)
        self.centre_psf_filename = None     # Filename of special PSF to be used for centroiding.
                                            # Used to ensure consistent PSF deconvolution
                                            # centroiding when producing PSFs
        self.source_window_radius = 30  # Radius of window [in pixels] centered on centre of
                                        # frame where source must be found for frame to not
                                        # be flagged as source-less
        self.centre_off_x = -0.710   # Typical target location offset from frame centre [pixels]
        self.centre_off_y = 1.055    # Typical target location offset from frame centre [pixels]
        self.mask_badpix = True  # True if bad pixels are to be masked
        self.mask_level = 2      # Mask out all before: 0 none, -2 dead pixels, -1 half-dead pixels, 
                                 # 3 telegraphic pixels, 2 saturated pixels, 1 hot pixels
        self.smear_corr = True   # True if smearing is to be corrected
        self.smear_resid_sa = False  # True if residual smearing effect is to be removed
        self.smear_resid_im = False # True if residual smearing effect is to be removed
        self.remove_static = True   # True if median of residual cube ("static") is to 
                                    # be subtracted (gets rid of dark current)
        self.pos_static = False  # Only subtract positive static image
        self.flatfield = True    # True if flat field correction should be applied
        self.darksub = True      # True if dark current is to be subtracted (from
                                 # Dark folder in calibpath)
        self.dark_min_snr = 15   # Minimum signal-to-noise ratio for dark estimate to include pixel.
                                 # This is set to avoid adding noise when subtracting dark.
        self.dark_min_level = 3.0 # [e-/second] Only correct for dark current pixels
                                  # above this level (to avoid low-level influence from
                                  # residual stars in dark)
        self.mask_bad_dark = True # Identify pixels with poor dark correction and mask them
        self.non_lin = True       # Apply non-linear correction if True

        # Charge transfer inefficiency parameters
        self.cti_corr = True       # True if CTI is to be corrected
        self.cti_t0 = 58800.0      # Zero epoch for CTI deterioration
        self.cti_scale = 0.0016    # CTI scaling paramter
        self.cti_expo = -0.65      # CTI exponent
        self.cti_lim = 0.0333      # Limiting CTI

        # Binary parameters
        self.secondary = 1       # Entry of secondary in starcat (primary is
                                 # always entry 0)
        self.psflib0 = None      # Binaries can have separate PSF libraries
        self.psflib1 = None      # defined for the two components
        self.init_flux_ratio = None  # Initial flux ratio estimate between 
                                     # binary components. If left to None, 
                                     # ratio will be read from starcat
        self.fix_flux2 = True        # Fix the flux of the secondary in a
                                     # second iteration to improve stability
        self.robust_centre_binary = False # In case of a close bright companion, 
                                          # a more robust but potentially less precise
                                          # centring algorithm can be attempted
        # Save switches
        self.save_mask_cube = True   # Save mask used to filter out bad
                                     # data (as fits file)
        self.save_bg_mask_cube = True  # If BG star mask defined, save it as fits
        self.save_resid_cube = True  # Save cube of residuals (as fits file)
        self.save_bg_cube = False    # Save cube of residuals with bg stars (as fits file)
        self.save_bg_models = False  # Save model of background, incl stars, smearing, static
        self.save_static = False     # Save derived static image as fits file
        self.save_psfmodel = False   # Save fitted model of PSF cube
        self.save_psf_list = True    # Save list of filenames of PSFs used
        self.save_psf_pc = False     # Save image cube of PSF principal components
        self.save_motion_mat = False # Save fitted motion blur matrix
        self.save_noise_cubes = False # Save estimated noise (raw/PSF/empiric) as fits cubes
        self.save_gain = False       # Save estimated gain table (with columns MJD, gain)
        self.save_bg_star_phot = True # If BG stars are fitted, save their photometry
        self.save_astrometry = False # For binaries, saves text file with separation
        
        # Extraction parameters
        self.klip = 5            # Number of eigen components of PSF to use.
        self.sigma_clip = 15     # The residual/std-factor for masking
        self.sigma_clip_niter = 2    # Number of iterations used for sigma-clipping
        self.empiric_noise = True   # Use noise determined by statistics on residuals
        self.empiric_sigma_clip = 4  # The sigma-clipping to use with empiric noise
        self.block_psf_level = 1e-4  # The level above which the PSF is blocked when
                                     # doing vertical smear correction
        self.centfit_rad = 23    # Find target and fit centroid inside this radius
        self.centfit_subrad = 3  # Compute flux centroid with this radius in deconvolved image
        self.motion_step = 0.3   # Step in fitting for motion blur
        self.motion_nsteps = 3   # Number of steps in each direction when
                                 # fitting for motion blur
        self.non_neg_lsq = False # Use non-negative least-square for motion blur
        
        self.smear_fact = 5.65       # Factor to multiply smear with to get the proper correction
        self.smear_const = 2e-7      # Calibrated from 55 Cnc observations (0.1 * exptime / ro_feq)
        self.ccdsize = (1024,1024)   # Size of full detector in pixels. Used for
                                     # smearing correction.
        self.pixel_scale = 1.01      # CHEOPS pixel scale in arcsec/pix

        self.sa_psfrad = 105      # Radius of area to subtract PSF in subarrays
        self.fitrad = 30         # Fit PSF inside this radius for flux
        self.normrad = 25        # Flux normalisation radius for PSF

        # Optimisation parameters. NB: time consuming.
        self.optimise_tree_iter = 5                 # Number of iterations in the search tree
        self.optimise_restarts = 3                  # Number of restarts for search
        self.sa_optimise = False                    # Optimise over range of parameters
                                                    # below -- only if set to True
        self.sa_test_klips = [1,3,5,10]             # Klip values to be tested
        self.sa_test_fitrads = [25,30,40,50,60]  # fitrad values to be tested
        self.sa_test_BG = True                      # Vary fit background?
        self.sa_test_Dark = True                    # Vary subtract dark?
        self.sa_test_Stat = True                    # Vary subtract static residual?

        self.im_optimise = False                    # Optimise over range of parameters
                                                    # below -- only if set to True
        self.im_test_klips = [1,3,5,10]             # Klip values to be tested
        self.im_test_fitrads = [25,30]              # fitrad values to be tested
        self.im_test_BG = True                      # Vary fit background?
        self.im_test_Dark = True                    # Vary subtract dark?
        self.im_test_Stat = True                    # Vary subtract static residual?


    def str_list(self):
        """ Returns list of strings for all parameters, where each string states
            the current value of a parameter
        """
        pl = []
        for k, v in self.__dict__.items():
            pl.append('{:s} = {}'.format( k, v))
        return pl
    
    def print_params(self):
        """ Prints the current values of all parameters
        """
        pl = self.str_list()
        for entry in pl:
            print(entry)
            
    def find_version(self, limit=10000):
        """ Finds unique version number of output files. Checks
        highest existing version in output directory and adds 1.
        """
        for version in range(limit):
            outdir = os.path.join(self.datapath, 'Outdata', '{:05d}'.format(version))
            if not os.path.isdir(outdir):
                return version
        raise Exception('Could not find available output directory')
            
    def define_filenames(self):
        """Finds and stores the filenames of relevant files in the datapath
        """
        # Helper function to parse files. Allows for alternative names
        # of files to be used, picks first match.
        def find_file(substrings, datapath=self.datapath, warn=True):
            for substring in substrings:
                for file in os.listdir(datapath):
                    if substring in file:
                        return os.path.join(datapath, file)
            if warn:
                print('Warning: \"{:s}\" file not found'.format(substrings[0]))
            return None

        # These are data files used by PIPE.
        self.file_att = find_file(("SCI_RAW_Attitude", "attitude."))
        self.file_sa_raw = find_file(("RAW_SubArray", "raw."))
        self.file_hk = find_file(("SCI_RAW_HkExtended",))
        self.file_im = find_file(("SCI_RAW_Imagette", "imagettes."), warn=False)
        self.file_starcat = find_file(("EXT_PRE_StarCatalogue", "starcat."))

        # Reference files
        self.file_gain = find_file(("REF_APP_GainCorrection", "gain."), 
                                   datapath=self.calibpath)
        self.file_nonlin = find_file(("nonlin.npy",), datapath=self.calibpath)
        self.file_flats = find_file(("REF_APP_FlatFieldTeff", "flats.fits",), datapath=self.calibpath)
        