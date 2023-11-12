# -*- coding: utf-8 -*-sigma_cl
"""
Created on Thu Jun  4 22:17:04 2020

@author: Alexis Brandeker, alexis@astro.su.se

Module that produces a library of principal PSF components from
a library of derived PSFs approximated by splines.

"""
import numpy as np
from .reduce import aperture
from scipy.linalg import eigh
from scipy.interpolate import BivariateSpline
from .psf_model import psf_model
import copy


class SplinePCA:
    """Class defining methods that manipulate a spline PSF library
    and produces a library of principal PSF components by eigen 
    decomposition of the spline coefficients, making use of the fact
    that splines are linear in their coefficients.
    """
    def __init__(self, psf_lib, radius=50, num_eigen=None):
        """psf_lib is a list of 2D splines defining PSFs, 
        radius is the radius out to which they are valid.
        num_eigen is the number of principal components to 
        be extracted and is defaulted to the maximum possible
        (the length of the psf_lib)
        """
        psf_matrix = make_psf_matrix(normalise_psf(psf_lib, radius=radius))
        self.median = np.median(psf_matrix, axis=0)
        self.tck0 = psf_lib[0].tck[0]
        self.tck1 = psf_lib[0].tck[1]
        self.degrees = psf_lib[0].degrees
        self.radius = radius
        self.median /= phot(self.get_median_spline())
        self.psf_matrix = psf_matrix - self.median
        self.update_eigen_lib(num_eigen)

    def update_eigen_lib(self, num_eigen=None):
        """Decompose the PSF library into eigen functions
        and compute normalisation coefficients for each fucntion.
        """
        self.eival, self.eivec = self.make_eigen_lib(num_eigen=num_eigen)
        self.einorm = np.zeros_like(self.eival)
        for n in range(len(self.eival)):
            self.einorm[n] = phot(self.get_eigen_spline(n))
    
    def make_eigen_lib(self, num_eigen=None):
        """Decompose the PSF library to the num_eigen first
        principal components, and return both the eigen values and
        the eigen vectors
        """
        T = self.psf_matrix
        num_refs = len(T)
        first_eigen = 0
        if num_eigen is not None:
            if num_eigen < num_refs:
                first_eigen = num_refs - num_eigen
        eigen_range = (first_eigen, num_refs - 1)
        eival, eivec = eigh(np.matmul(T, T.T), eigvals = eigen_range) 
        base_vec = np.flip(np.matmul(T.T, eivec), axis=1)
        base_vec /= np.sum(base_vec**2, axis=0)**0.5
        return np.flip(eival), base_vec.T
    
    def get_eigen_spline_lib(self, klip=None):
        """Produces a list of  the klip first splines out of the
        list of principal component coefficients
        """
        if klip is None:
            klip = len(self.eivec)
        eig_spl_lib = []
        for n in range(-1, klip):
            eig_spl_lib.append(self.get_eigen_spline(n))
        return eig_spl_lib
    
    def get_median_spline(self):
        """Returns the spline of median PSF library coefficients
        """
        return self.get_spline(self.median)
  
    def get_eigen_spline(self, index):
        """Return the spline corresponding to principal component index
        """
        if index == -1:
            return self.get_median_spline()
        return self.get_spline(self.eivec[index,])
    
    def get_spline(self, spl_coeff):
        """Given a set of spline coefficients spl_coeff, produce 
        and return the corresponding 2D spline
        """
        return make_spline2D((self.tck0, self.tck1, spl_coeff, self.degrees[0], self.degrees[1]))
   

def make_psf_matrix(psf_lib):
    """Given a list of PSF splines (psf_lib), produce a matrix of their
    spline coefficients. The splines are assumed to have equal number
    of coefficients.
    """
    psf_matrix = np.zeros((len(psf_lib), len(psf_lib[0].get_coeffs())))
    for n in range(len(psf_lib)):
        psf_matrix[n, :] = psf_lib[n].get_coeffs()
    return psf_matrix


def make_spline2D(spl_params):
    """Make a 2D spline out of its defining parameters
    The paramaters are (tck0, tck1, tck3, degree0, degree1)
    used by BiVariateSpline to define a spline
    """
    return BivariateSpline()._from_tck(spl_params)


#def make_spline2D(degrees, tck):
#    """Make a 2D spline out of its defining parameters
#    """
#    spl = BivariateSpline()
#    spl.degrees = degrees
#    spl.tck = copy.deepcopy(tck)
#    return spl


def sum_spline(psf_lib, weights=None):
    """Produce the sum of a list of splines by adding their coefficients.
    If weights is defined, a weighted sum is produced. The summed spline
    is returned.
    """
    psf_matrix = make_psf_matrix(psf_lib)
    if weights is not None:
        psf_matrix *= weights[:,None]
    coeffs = np.sum(psf_matrix, axis=0)
    return make_spline2D((psf_lib[0].tck[0], psf_lib[0].tck[1], coeffs,
                          psf_lib[0].degrees[0] , psf_lib[0].degrees[1]))
    

def psf_coeff(psf_fun):
    """Return the spline coefficients for a PSF
    """
    return psf_fun.tck[2].copy()


def median_psf(psf_lib):
    """Compute the median of the coefficients of a list of PSF splines,
    and return a spline with the median coefficients.
    """
    mat = make_psf_matrix(psf_lib)
    spl_params = (psf_lib[0].tck[0], psf_lib[0].tck[1], np.median(mat, axis=0),
                  psf_lib[0].degrees[0], psf_lib[0].degrees[1])
    return make_spline2D(spl_params)
    

def normalise_psf(psf_lib, radius=50):
    """Given a list of PSF splines in psf_lib, return a list of the
    same splines normalised by their integral out to radius.
    """
    norms = psf_integral(psf_lib, radius)
    ret_lib = copy.deepcopy(psf_lib)
    for n in range(len(ret_lib)):
        ret_lib[n].tck= (psf_lib[n].tck[0], psf_lib[n].tck[1], psf_lib[n].tck[2]/norms[n])
    return ret_lib


def psf_integral(psf_lib, radius=50):
    """Return a list of the integral of each PSF spline listed
    in psf_lib, out to radius.
    """
    norms = np.zeros(len(psf_lib))
    for n in range(len(psf_lib)):
        norms[n] = phot(psf_lib[n], radius=radius)
    return norms


def phot(psf_fun, radius=50, sample=10):
    """Integrate PSF spline function over circular region
    defined by radius
    """
    pm = psf_model(psf_fun, norm_rad=radius, sample=sample)
    return pm.norm


if __name__=='__main__':
    print('Do not run')
