# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 23:35:43 2020

@author: Alexis Brandeker, alexis@astro.su.se

Containes a class that derives PSFs from data contained in a PsfPhot object.
Multi-threaded and calls the psf_worker module.

"""

import numpy as np
import multiprocessing as mp
import pickle
from .psf_worker import make_psf
from .reduce import check_val, aperture
from .analyse import find_orbits


class MultiPSFMaker():
    """ Collects routines to compute the PSF from data contained in 
    the PsfPhot object pp. 
    """
    def __init__(self, pp, outrad=70, max_threads=8):
        """ pp is a PsfPhot object
            outrad is the radius out to which the PSF will be defined
            max_thread is the maximum number of threads that will
            be used in the PSF production pool. Actual number of
            threads may be smaller, depedning on how many PSFs are
            to be produced.
        """
        self.pp = pp
        self.outrad = outrad
        self.psf_spline = pp.psf
        self.sa_mask_org = self.pp.sa_mask.copy()
        self.im = pp.pps.file_im is not None
        if self.im is False:
            self.pp.plog('MPM - WARNING: no imagettes defined')
        self.measure_flux()
        self.select_frames()
        self.max_threads = max_threads


    def measure_flux(self, radius=23):
        """Do crude photometry of all frames, in order to later normalise
        and combine them for PSF estimates.
        """
        self.sa_flux = self.pp.apt_phot_sa(radius=radius)
        if self.im:
            self.im_flux = self.pp.apt_phot_im(radius=radius)
    
    
    def select_frames(self, clip=3, clip_niter=10):
        """Select frames that are good for estimating PSF
        """
        if self.im:
            sel, ret_val = check_val(self.im_flux, clip, clip_niter)
            sel *= self.pp.filter_motion(self.pp.im_xc, self.pp.im_yc, lowfrac=0.9)
            sel *= self.pp.filter_pos(self.pp.im_xc, self.pp.im_yc)
            sel *= self.pp.filter_bad_masks(self.pp.im_mask_cube, self.pp.im_apt, clip=10)
            self.im_sel = sel
        sel, ret_val = check_val(self.sa_flux, clip, clip_niter)
        sel *= self.pp.filter_motion(self.pp.sa_xc, self.pp.sa_yc, lowfrac=0.99)
        sel *= self.pp.filter_pos(self.pp.sa_xc, self.pp.sa_yc)
        sel *= self.pp.filter_bad_masks(self.pp.sa_mask_cube, self.pp.sa_apt, clip=10)
        self.sa_sel = sel
 
    
    def find_ranges(self, phase=None, sub_orbits=1):
        """Find ranges of subarray frames to be used for deriving independent
        PSFs. Default is to derive 1 PSF per orbit, but this can be changed by
        the number of sub_orbits (i.e. how many PSFs per orbit should be derived).
        The roll angle phase can be defined for the start of an orbit. A list
        of range pairs is returned.
        """
        orbs = find_orbits(self.pp.sa_att[:,3], phase=phase, sub_orbits=sub_orbits)
        return [(orbs[n],orbs[n+1]) for n in range(len(orbs)-1)]        
        
            
    def make_pixtab(self, clip=3, sa_range = None, limit=0.5):
        """Produce pixel table from subarray frames and imagettes,
        filter out frames inappropriate for PSF derivation. We can afford
        to be picky and only accept good frames. If the number of skipped
        frames in a range is larger than limit, no pixel table is produced.
        """
        sa_sel = self.sa_sel.copy()

        if (np.sum(sa_sel[sa_range[0]:sa_range[1]]) /
            len(sa_sel[sa_range[0]:sa_range[1]])) < limit:
            return None

        if sa_range is not None:
            sa_sel[:sa_range[0]] = 0
            sa_sel[sa_range[1]:] = 0

        pix_sa = self.pp.make_pixtab_sa(sa_sel, self.pp.sa_mask,
                                        self.sa_flux[sa_sel])
        self.pp.mess('MPM - SA: {:.2f} Mpix'.format(len(pix_sa)/1e6))
    
        # Filter out inappropriate frames
        if self.im:
            im_sel = self.im_sel.copy()
            if sa_range is not None:
                im_range = (int(sa_range[0]*self.pp.nexp), 
                            int(sa_range[1]*self.pp.nexp))
                im_sel[:im_range[0]] = 0
                im_sel[im_range[1]:] = 0
            pix_im = self.pp.make_pixtab_im(im_sel, self.pp.im_mask,
                                            self.im_flux[im_sel])
            self.pp.mess('MPM - IM: {:.2f} Mpix'.format(len(pix_im)/1e6))

        if self.im:
            pixtab = np.concatenate((pix_im, pix_sa), axis=0)
        else:
            pixtab = pix_sa
        sel = np.abs(pixtab[:,0]) <= (self.outrad+1)
        sel *= np.abs(pixtab[:,1]) <= (self.outrad+1)
        return pixtab[sel,:]


    def prod_psf(self, sa_ranges, lib_num):
        """Produce a PSF for each subarray frame range in the sa_ranges
        list. Save pickled PSF list to file enumerated with lib_num. 
        """
        self.pp.mess('MPM - Producing PSFs ({:d})'.format(lib_num))
        self.pixtabs = []
        nthreads = min(max(len(sa_ranges), 1), self.max_threads)
        
        for n in range(len(sa_ranges)):
            self.pp.mess(f'MPM - Making pix table, range {sa_ranges[n]}')

            pixtab = self.make_pixtab(sa_range=sa_ranges[n])
            if pixtab is not None:
                self.pixtabs.append(pixtab)
        
        if self.psf_spline is None:
            self.pp.mess('MPM - PSF not found - defining new PSF...')
            self.psf_spline = make_psf(self.pixtabs[0], self.outrad,
                                                   polydeg=2, niter=1)

        self.pp.mess('MPM - Updating mask...')
        badmask = update_sa_mask(self.psf_spline, self.pp.sa_sub[self.sa_sel],
                                 self.sa_flux[self.sa_sel],
                                 self.pp.sa_xc[self.sa_sel],
                                 self.pp.sa_yc[self.sa_sel],
                                 self.outrad)
        self.badmask = badmask * self.pp.sa_mask
        self.pp.mess('MPM - Badmask update: {:d} new bad pixels ({:d} from before, {:d} total)'.format(
                np.sum(badmask==0),
                np.sum(self.pp.sa_mask==0),
                np.sum(self.badmask==0)))

        self.pp.mess(f'MPM - Making PSFs, {nthreads} threads')
        with mp.Pool(nthreads) as p:
            psf_lib = p.starmap(make_psf, [(pixtab, self.outrad) for
                                                 pixtab in self.pixtabs])
        self.save_psf(lib_num, psf_lib)
        self.pp.mess('MPM - Done!')

        return psf_lib


    def save_psf(self, lib_num, psf_lib, component='A'):
        """Save a pickled file of the PSF list psf_lib. Lib_num is
        an arbitrary version number used in the filename.
        """
        filename = self.pp.psf_name(lib_num, component=component)
        self.pp.mess(f'Saving PSF to \'{filename}\'')        
        with open(filename, 'wb') as fp:
            pickle.dump(psf_lib, fp)



def update_sa_mask(psf_spline, data, flux, xc, yc, outrad, bad_pix_frac=0.01):
    """Derive a new bad pixel map, from statistics on PSF-subtracted subarrays.
    """
    low_quant = 0.2
    res = np.zeros_like(data)
    for n in range(len(data)):
        xcoo = np.arange(data.shape[1])-xc[n]
        ycoo = np.arange(data.shape[2])-yc[n]
        res[n] = data[n]-flux[n]*psf_spline(ycoo,xcoo)
    low = np.quantile(res, low_quant, axis=0)
    apt_out = aperture(low.shape, radius=outrad)
    bad_pix = (low > np.quantile(low[apt_out], 1-bad_pix_frac))*apt_out
    return (bad_pix == 0)


if __name__=='__main__':
    print('Do not run!')

