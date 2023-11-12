# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:32:21 2022

@author: Alexis Brandeker

Defines a class to contain a PSF model 2D-spline, handling transformation between
regular detector coordinates and their non-linear "stretching" of the spline coordinates.
This stretching is applied to have higher spline resolution near the core of the PSF and
lower in the wings. The stretching function "radial_map" has a hardcoded stretching
function that is globally used valid for pixel radii up to 200 pix.
"""

import numpy as np
from scipy import interpolate

def radial_map(sample_points=40, int_fun='linear'):
    """Defines a non-linear map and its inverse of radial pixel
    coordinates. This radial map is used to define PSFs on
    a non-linear grid, such that the wings of the PSF are sampled 
    less densely than the core. Function returns two 1D-interpolation
    functions for_map and inv_map, such that inv_map(for_map(x)) == x 
    (approximately). Valid out to radii of 200 pix (corresponding to
    40 pix in PSF spline coordinates)
    """
    def_x = np.array([0, 20, 50, 100, 200]) # Defines the non-linear law
    def_y = np.array([0, 20, 30, 35, 40])
    t = np.linspace(0, 200**.5, sample_points)**2
    
    f = interpolate.interp1d(def_x, def_y, kind='quadratic',
                             fill_value='extrapolate')
    for_map = interpolate.interp1d(t, f(t), kind=int_fun,
                                   fill_value='extrapolate')
    inv_map = interpolate.interp1d(f(t), t, kind=int_fun,
                                   fill_value='extrapolate')
    return for_map, inv_map


class map_coo():
    """Transforms between detector coordinates and coordinates used
    by the PSF spline ("PSF coordinates")
    """
    def __init__(self, sample_points=40):
        self.for_map, self.inv_map = radial_map(sample_points)
    
    def to_detect(self, x, y):
        """Maps coordinates in PSF plane to
        detector plane
        """
        r = (x**2 + y**2)**.5
        s = self.inv_map(r)/np.maximum(r, 1e-3)
        return s*x, s*y
    
    def to_psf(self, x, y):
        """Maps coordinates in detector plane to
        PSF plane
        """
        r = (x**2 + y**2)**.5
        s = self.for_map(r)/np.maximum(r, 1e-3)
        return s*x, s*y
    
    def pixtab_to_psf(self, pixtab):
        out_pixtab = pixtab.copy()
        out_pixtab[:,0], out_pixtab[:,1] = self.to_psf(pixtab[:,0], pixtab[:,1])
        return out_pixtab
    

class psf_model():
    """A PSF model that uses a 2D spline but with a mapping between
    coordinates used by the spline and coordinates of pixels on the
    detector. This is to enable a non-linear resolution of the PSF
    spline, i.e. position of the knots, in detector coordinates.
    """
    def __init__(self, psf_spl, norm_rad=50, sample=10):
        self.map = map_coo()
        self.psf_spl = psf_spl
        self.norm = self.norm_psf(radius=norm_rad, sample=sample)

    def norm_psf(self, radius, sample):
        x = np.linspace(-radius, radius, 2*radius*sample+1)
        psf_frame = self.__call__(x, x, circular=True)
        return np.sum(psf_frame)/sample**2

    def __call__(self, x, y, grid=True, circular=False):
        if grid:
            xx, yy = np.meshgrid(x, y)
        else:
            xx, yy = x, y
        px, py = self.map.to_psf(xx, yy)
        if circular:
            support = xx**2+yy**2 <= min(np.max(x**2), np.max(y**2))
            return support*self.psf_spl(px, py, grid=False)
        return self.psf_spl(px, py, grid=False)
    
    def get_coeffs(self):
        return self.psf_spl.get_coeffs()