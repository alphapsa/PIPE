# -*- coding: utf-8 -*-
"""
Created on Fri Apr 05 18:42:00 2024

@author: Alexis Brandeker, alexis@astro.su.se

Define non-linearity corrections. 

"""

def non_lin_tweak(flux, nexp=1, params=(100, 700, 0.04, 800)):
    """flux is pixel values in electrons. nexp is the number of co-added
    exposures, used to determine the actual exposure per frame. This
    function produces a correction to the non-linear function. Divide
    the pixel values (in electrons) by the return of this function for
    the correction.
    """
    c0, c1, c2, c3 = params
    ret = (flux/nexp - c0)/c1*c2 + (1-c2)
    ret[flux/nexp > c3] = 1
    return ret
