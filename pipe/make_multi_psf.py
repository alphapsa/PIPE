# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 23:35:43 2020

@author: Alexis Brandeker, alexis@astro.su.se

Containes a class that derives PSFs from data contained in a PsfPhot object.
Multi-threaded and calls the psf_worker module.

"""

import numpy as np
import multiprocessing as mp
from .psf_worker import make_psf_spline
from .reduce import check_val, aperture
from .analyse import find_orbits
from .psf_model import map_coo, psf_model
from .pipe_statistics import sigma_clip

try:
   mp.set_start_method('spawn', force=True)
except RuntimeError:
   pass


class MultiPSFMaker:
    """ Collects routines to compute the PSF from data contained in 
    the PsfPhot object pp. 
    """
    def __init__(self, pp, outrad=100, max_threads=8):
        """ pp is a PsfPhot object
            outrad is the radius out to which the PSF will be defined
            max_thread is the maximum number of threads that will
            be used in the PSF production pool. Actual number of
            threads may be smaller, depedning on how many PSFs are
            to be produced.
        """
        self.pp = pp
        self.map_coo = map_coo()
        self.outrad = outrad
        self.psf_outrad = self.map_coo.for_map(self.outrad)
        self.psf_mod = pp.psf
        self.im = pp.pps.file_im is not None
        self.sa_mask = self.pp.sa_mask.copy()
        if self.im:
            self.im_mask = self.pp.im_mask.copy()
        else:
            self.pp.mess('MPM - WARNING: no imagettes defined')

        self.margin = 1.02  # Used for mocking pixels in margin for a more robust fit
        self.oversample = 3 # Mocked pixels per spline knot
        self.assign_flux()
        self.select_frames()
        self.max_threads = max_threads


    def assign_flux(self, radius=23):
        """Do crude photometry of all frames, in order to later normalise
        and combine them for PSF estimates.
        """
        self.sa_flux = self.pp.sa_flux
        if self.im:
            self.im_flux = self.pp.im_flux
    
    
    def select_frames(self, clip=3, clip_niter=10):
        """Select frames that are good for estimating PSF
        """
        if self.im:
            sel, _ret_val = check_val(self.im_flux, clip, clip_niter)
            sel *= self.pp.filter_motion(self.pp.im_xc, self.pp.im_yc, lowfrac=0.9)
            sel *= self.pp.filter_pos(self.pp.im_xc, self.pp.im_yc)
            sel *= self.pp.filter_bad_masks(self.pp.im_mask_cube, self.pp.im_apt, clip=10)
            sel *= self.pp.im_sel
            self.im_sel = sel 
        sel, _ret_val = check_val(self.sa_flux, clip, clip_niter)
        sel *= self.pp.filter_motion(self.pp.sa_xc, self.pp.sa_yc, lowfrac=0.99)
        sel *= self.pp.filter_pos(self.pp.sa_xc, self.pp.sa_yc)
        sel *= self.pp.filter_bad_masks(self.pp.sa_mask_cube, self.pp.sa_apt, clip=10)
        sel *= self.pp.sa_sel
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
        
            
    def make_psf_pixtab(self, sa_range = None, limit=0.5):
        """Produce pixel table from subarray frames and imagettes,
        filter out frames inappropriate for PSF derivation. We can afford
        to be picky and only accept good frames. If the number of skipped
        frames in a range is larger than limit, no pixel table is produced.
        """
        sa_sel = self.sa_sel.copy()
        N_sel = np.sum(sa_sel[sa_range[0]:sa_range[1]])
        N_tot = len(sa_sel[sa_range[0]:sa_range[1]])

        if (N_sel / N_tot) < limit:
            self.pp.mess('MPM - too many skipped frames ({:d}/{:d})'.format(N_sel, N_tot))
            return None

        if sa_range is not None:
            sa_sel[:sa_range[0]] = 0
            sa_sel[sa_range[1]:] = 0

        pix_sa = self.pp.make_pixtab_sa(sa_sel, self.sa_mask,
                                        self.sa_flux[sa_sel])
        self.pp.mess('MPM - SA: {:.2f} Mpix'.format(len(pix_sa)/1e6))
    
        if self.im:
            im_sel = self.im_sel.copy()
            if sa_range is not None:
                im_range = (int(sa_range[0]*self.pp.nexp), 
                            int(sa_range[1]*self.pp.nexp))
                im_sel[:im_range[0]] = 0
                im_sel[im_range[1]:] = 0
            pix_im = self.pp.make_pixtab_im(im_sel, self.im_mask,
                                            self.im_flux[im_sel])
            pix_sa[:,3] *= self.pp.nexp**.5 # Devalue contribution from SA,
                                            # makes IM more important for core PSF
            self.pp.mess('MPM - IM: {:.2f} Mpix'.format(len(pix_im)/1e6))

        if self.im:
            pixtab = np.concatenate((pix_im, pix_sa), axis=0)
        else:
            pixtab = pix_sa
        sel = np.abs(pixtab[:,0]) <= (self.outrad+1)
        sel *= np.abs(pixtab[:,1]) <= (self.outrad+1)
        new_pixtab = self.add_border_pixels(pixtab[sel,:])
        psf_pixtab = self.fill_psf_pixels(self.map_coo.pixtab_to_psf(new_pixtab))

        return psf_pixtab


    def add_border_pixels(self, pixtab, value=1e-7, error_fact=5):
        """In the pixel table, add pixels of value around the border
        to fit a spline more graciously to the PSF.
        error_fact is the factor to multiply the median error for assigning
        the new pixels.
        """
        outer = self.outrad*self.margin
        v = np.linspace(-outer, outer, int(self.oversample*self.outrad))
        xx, yy = np.meshgrid(v, v)
        r = (xx**2+yy**2)**0.5
        values = np.zeros_like(r)
        errors = np.ones_like(r)*np.median(pixtab[:,3])*error_fact
        sel = (r <= outer)*(r >= (self.outrad-1))
        values[sel] = value
        num_new_pix = int(np.sum(sel))
        ntabrows, ntabcols = pixtab.shape
        new_pixtab = np.zeros(((ntabrows+num_new_pix), ntabcols))
        new_pixtab[:ntabrows, :ntabcols] = pixtab
        new_pixtab[ntabrows:, 0] = xx[sel]
        new_pixtab[ntabrows:, 1] = yy[sel]
        new_pixtab[ntabrows:, 2] = values[sel]
        new_pixtab[ntabrows:, 3] = errors[sel]
        return new_pixtab
        

    def fill_psf_pixels(self, pixtab, error_fact=5):
        """In the pixel table, fill out the corners outside radius of the
        PSF fitting areas with pixels of value zero. This because the 
        LSQBivariateSpline does not accept non-rectangular support, while
        the valid pixels in a subarray are confined to a circle.
        error_fact is the factor to multiply the median error for assigning
        the new pixels.
        """
        outer = self.psf_outrad * self.margin
        v = np.linspace(-outer, outer, int(self.oversample*self.outrad))
        xx, yy = np.meshgrid(v, v)
        r = (xx**2+yy**2)**0.5
        values = np.zeros_like(r)
        errors = np.ones_like(r)*np.median(pixtab[:,3])*error_fact
        sel = self.map_coo.inv_map(r) >= self.outrad
        num_new_pix = int(np.sum(sel))
        ntabrows, ntabcols = pixtab.shape
        new_pixtab = np.zeros(((ntabrows+num_new_pix), ntabcols))
        new_pixtab[:ntabrows, :ntabcols] = pixtab
        new_pixtab[ntabrows:, 0] = xx[sel]
        new_pixtab[ntabrows:, 1] = yy[sel]
        new_pixtab[ntabrows:, 2] = values[sel]
        new_pixtab[ntabrows:, 3] = errors[sel]
        return new_pixtab


    def prod_psf(self, sa_ranges, outdir=None):
        """Produce a PSF for each subarray frame range in the sa_ranges
        list. Save pickled PSF list to file. 
        """
        self.pp.mess('MPM - Producing PSFs')
        self.pixtabs = []
        MJD = []
        TF2 = []
        nthreads = min(max(len(sa_ranges), 1), self.max_threads)

        if self.im:
            exptime = self.pp.im_hdr['EXPTIME'] * self.pp.im_nexp
        else:
            exptime = self.pp.sa_hdr['TEXPTIME']

        for n in range(len(sa_ranges)):
            self.pp.mess(f'MPM - Making pix table, range {sa_ranges[n]}')

            pixtab = self.make_psf_pixtab(sa_range=sa_ranges[n])
            if pixtab is not None:
                self.pixtabs.append(pixtab)
                n0, n1 = sa_ranges[n][0], sa_ranges[n][1]
                MJD.append(np.mean(self.pp.sa_mjd[n0:n1]))
                ind = sigma_clip(self.pp.sa_thermFront_2[n0:n1])
                TF2.append(np.mean(self.pp.sa_thermFront_2[n0:n1][ind]))
        
        if self.psf_mod is None:
            self.pp.mess('MPM - PSF not found - defining new PSF...')
            self.psf_mod = psf_model(make_psf_spline(self.pixtabs[0], self.psf_outrad,
                                                   polydeg=2, niter=1))

        self.pp.mess('MPM - Updating mask...')
        sa_bad = update_sa_mask(self.psf_mod, self.pp.sa_sub[self.sa_sel],
                                 self.sa_flux[self.sa_sel],
                                 self.pp.sa_xc[self.sa_sel],
                                 self.pp.sa_yc[self.sa_sel],
                                 self.outrad)
        self.sa_mask = sa_bad * self.pp.sa_mask
        self.pp.mess('MPM - Badmask update: {:d} new bad pixels ({:d} from before, {:d} total)'.format(
                np.sum(sa_bad==0),
                np.sum(self.pp.sa_mask==0),
                np.sum(self.sa_mask==0)))

        self.pp.mess('MPM - Badmask update: {:d} new bad pixels ({:d} from before, {:d} total)'.format(
                np.sum(sa_bad==0),
                np.sum(self.pp.sa_mask==0),
                np.sum(self.sa_mask==0)))


        self.pp.mess(f'MPM - Making PSFs, {nthreads} threads')
        with mp.Pool(nthreads) as p:
            psf_lib = p.starmap(make_psf_spline, [(pixtab, self.psf_outrad) for
                                                 pixtab in self.pixtabs])
        self.save_psf(psf_lib=psf_lib,
                      xc=self.pp.psf_lib.target_x,
                      yc=self.pp.psf_lib.target_y,
                      Teff=self.pp.pps.Teff,
                      TF2=TF2, MJD=MJD,
                      exptime=exptime,
                      outdir=outdir)
        self.pp.mess('MPM - Done!')
        return psf_lib


    def save_psf(self, psf_lib, xc, yc, Teff, TF2, MJD, exptime, outdir=None):
        """Save PSF spline coefficients in a file for each PSF in the list psf_lib.
        Lib_num is an arbitrary version number used in the filename.
        """
        N = len(psf_lib)

        for n in range(N):
            filename = self.pp.psf_lib.filename(xc=xc, yc=yc,
                                                Teff=Teff, TF2=TF2[n],
                                                mjd=MJD[n],
                                                exptime=exptime,
                                                outdir=outdir)
            self.pp.mess(f'Saving PSF {n}/{N} to \'{filename}\'')        

            spl = psf_lib[n]
            tck = spl.tck
            deg = spl.degrees
            outdata = np.array((tck[0], tck[1], tck[2], deg[0], deg[1]), dtype=object)
            np.save(filename, outdata)


 
def update_sa_mask(psf_mod, data, flux, xc, yc, outrad, bad_pix_frac=0.01):
    """Derive a new bad pixel map, from statistics on PSF-subtracted subarrays.
    """
    low_quant = 0.2
    res = np.zeros_like(data)
    for n in range(len(data)):
        xcoo = np.arange(data.shape[2])-xc[n]
        ycoo = np.arange(data.shape[1])-yc[n]
        res[n] = data[n]-flux[n]*psf_mod(ycoo,xcoo)
    low = np.quantile(res, low_quant, axis=0)
    apt_out = aperture(low.shape, radius=outrad)
    bad_pix = (low > np.quantile(low[apt_out], 1-bad_pix_frac))*apt_out
    return (bad_pix == 0)


if __name__=='__main__':
    print('Do not run!')

