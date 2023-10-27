# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:36:13 2020

@author: Alexis Brandeker, alexis@astro.su.se

Module with the star_bg class and methods to use the catalog file of background
stars (retrieved from Gaia) to produce synthetic images of the field of view
observed by CHEOPS, using an empirical PSF.
"""
import numpy as np
from numpy.linalg import lstsq
from .reduce import coo_mat
from .syntstar import star_bg, def_psf_rad, find_area_inds



class star_bg():
    """Encapsulates stellar background fit data, such as
    fitted positions, brightness, and assigned PSF of background
    stars. Each star has a collection of PSF offsets defined to
    account for image blurring due to image rotation and 
    jittering.
    """
    def __init__(self, psf_ids, xlists, ylists, fscales, rscales=[]):
        """Input parameters consist of lists of lengths equal to the
        number of background stars considered.
        psf_ids the ID of the PSF to be used for the corresponding star
        Each entry in xlists and ylists is an np array in itself of 
        coordinates for offset centers of the PSF to be combined.
        fscale is the total flux scale for the star (typically
        expressed in units of the target, but not necessarily)
        Each entry of rscales, if existent, is an np array of
        relative brightness for the offset PSFs, with a total
        brightness normalised to 1.
        """
        self.psf_ids = psf_ids
        self.xlists = xlists
        self.ylists = ylists
        self.fscales = fscales
        self.rscales = rscales
        if self.rscales == []:
            for xlist in self.xlists:
                if len(xlist) > 0:
                    self.rscales.append(np.ones_like(xlist)/len(xlist))
                else:
                    self.rscales.append(np.empty(0))

    def get_psf_id(self, index):
        """Returns the PSF ID for star number index
        """
        return self.psf_ids[index]
        
    def image_psf(self, psf, index, shape, off_x, off_y, rscale=None):
        """Generate subimage with PSF from list of offsets
        (to take into account image blur and jitter)
        """
        xcoo = np.arange(shape[1]) - off_x
        ycoo = np.arange(shape[0]) - off_y
        ret_img = np.zeros(shape)
        psf_rad = def_psf_rad(self.fscales[index])

        if rscale is None:
            rscale = self.rscales[index]

        for x, y, rs in zip(self.xlists[index], self.ylists[index], rscale):
            ret_img += rs * psf(ycoo - y, xcoo - x)
        xmat, ymat = np.meshgrid(xcoo - np.mean(x), ycoo - np.mean(y))
        ret_img[xmat**2 + ymat**2 > psf_rad**2] = 0
        ret_img *= self.fscales[index]
        return ret_img

    def fit(self, psf, index, subimage, off_x, off_y, fit_rad=25):
        """Fit PSF to subimage using pre-defined collection of
        offsets for PSF.
        """
        psf_rad = def_psf_rad(self.fscales[index])
        xmat, ymat = coo_mat(subimage.shape, off_x, off_y)
        xcoo =  np.arange(subimage.shape[0]) - off_x
        ycoo =  np.arange(subimage.shape[1]) - off_y

        
