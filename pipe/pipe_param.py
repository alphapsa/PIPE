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
        self.ron = None          # read-out noise in RMS electrons per
                                 # readout and pixel; estimated if not defined
        self.bias = None         # bias in ADU; estimated if not defined
        self.sa_range = sa_range # tuple of 2 integers: Range of subarray indices to
                                 # be considered. If "None", the full array is used.
        self.mjd2bjd = True      # Use barycentric conversion from MJD to BJD
        self.binary = False      # If the binary code branch is to be used, requires
                                 # special parameters to be defined, see below
        self.psflib = None       # Name of library of PSF eigen functions to be used
                                 # (located in calibpath/psf_lib/)
        self.psf_rad = 70        # PSF lib defined out to this integer radius
        self.nthreads = os.cpu_count()-1    # Number of threads to use; defaulted to 
                                            # the number of system virtual cores - 1
        self.bg_fit = 0          # Simultaneous background to be fit with PSF:
                                 # -1: no, 0: constant, 1: bilinear plane, 2: parabolic 
        self.drp_bg_corr = True  # True if bg subtraction from DRP should be corrected
        self.resample_im_times = True   # True if the time stamps for imagettes should
                                        # be corrected (SOC bug) by interpolating
                                 # subarray times stamps
        self.bgstars = True      # True if background stars are to be subtracted
        self.limflux = 1e-3      # Inlude bgstars down to this fraction of the target flux
        self.mask_badpix = True  # True if bad pixels are to be masked
        self.smear_corr = True   # True if smearing is to be corrected
        self.smear_resid = False # True if residual smearing effect is to be removed
        self.remove_static = True    # True if median of residual cube ("static") is to 
                                     # be subtracted (gets rid of dark current)
        self.static_psf_rad = True   # Limit static subtraction to area defined by PSF
        self.flatfield = True    # True if flat field correction should be applied
        self.darksub = True      # True if dark current is to be subtracted (from
                                 # Dark folder in calibpath)
        self.dark_level = 2.0    # [e-/second] Only correct for dark current pixels
                                 # above this level (to avoid low-level influence from
                                 # residual stars in dark) 
        self.non_lin = True      # Apply non-linear correction to imagettes if True
        self.non_lin_tweak = False # Apply empirical tweak to non-linear correction
                                   # at low exposure levels, for both subarrays
                                   # and imagettes
        self.nl_100 = 0.04        # Non-linearity tweak correction at 100 electrons
        self.nl_lim = 800       # Non-linearity tweak until this number of electrons
        self.empiric_noise = False # Use noise estimated from PSF residuals in time
                                   # series rather than assuming photon statistics
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
        # Save switches
        self.save_mask_cube = True   # Save mask used to fitler out bad
                                     # data (as fits file)
        self.save_resid_cube = True  # Save cube of residuals (as fits file)
        self.save_static = False     # Save derived static image as fits file
        self.save_psfmodel = False   # Save fitted model of PSF cube
        self.save_motion_mat = False # Save fitted motion blur matrix
        self.save_noise_cubes = False # Save estimated noise (raw/PSF/empiric) as fits cubes
        
        # Extraction parameters
        self.klip = 10           # Number of eigen components of PSF to use.
        self.sigma_clip = 15     # The residual/std-factor for masking
        self.sigma_clip_niter = 2    # Number of iterations used for sigma-clipping
        self.empiric_noise = False   # Use noise determined by statistics on residuals
        self.empiric_sigma_clip = 5  # The sigma-clipping to use when using empiric noise
        self.block_psf_level = 1e-4  # The level above which the PSF is blocked when
                                     # doing vertical smear correction
        self.centfit = 23        # Fit PSF inside this radius for centroid
        self.fluxfit = 40        # Fit PSF inside this radius for flux
        self.motion_step = 0.3   # Step in fitting for motion blur
        self.motion_nsteps = 3   # Number of steps in each direction when
                                 # fitting for motion blur
        self.non_neg_lsq = False # Use non-negative least-square for motion blur
        
        self.smear_fact = 5.65       # Factor to multiply smear with to get the proper correction
        self.ccdsize = (1024,1024)   # Size of full detector in pixels. Used for
                                     # smearing correction.
        
        self.sa_psfrad = 50      # Radius of area to subtract PSF in subarrays
        self.fitrad = 30         # Use this radius for fitting PSF
        self.im_psfrad = 23      # Radius of area where most of the PSF flux is
                                 # Used for fitting and rough aperture photometry                                 

                                 
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
        # These are data files used by PIPE. Some are DRP output files.
        self.file_att = find_file(("SCI_RAW_Attitude", "attitude."))
        self.file_sa_cor = find_file(("SCI_COR_SubArray", "bgsub."))
        self.file_sa_cal = find_file(("SCI_CAL_SubArray", "biassub."))
        self.file_sa_raw = find_file(("RAW_SubArray", "raw."))
        self.file_hk = find_file(("SCI_RAW_HkExtended",))
        self.file_im = find_file(("SCI_RAW_Imagette", "imagettes."))
        self.file_mask = find_file(("PIP_COR_PixelFlagMapSubArray", "mask."))
        self.file_starcat = find_file(("EXT_PRE_StarCatalogue", "starcat."))
        self.file_lc_default = find_file(("SCI_COR_Lightcurve-DEFAULT",))
        self.file_lc_optimal = find_file(("SCI_COR_Lightcurve-OPTIMAL",))
        self.file_lc_rinf = find_file(("SCI_COR_Lightcurve-RINF",))
        self.file_lc_rsup = find_file(("SCI_COR_Lightcurve-RSUP",))
        # Reference files
        self.file_gain = find_file(("REF_APP_GainCorrection", "gain."), 
                                   datapath=self.calibpath)
        self.file_nonlin = find_file(("nonlin.txt",), datapath=self.calibpath)
        self.file_flats = find_file(("flats.fits",), datapath=self.calibpath)
        