# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:05:46 2023

@author: Alexis Brandeker

Module to generate a sequence of tests to find parameters that
generate lowest MAD-values for light curves. This is sort-of 
a gradient search: start with a vector of parameter types (e.g.
klip, radfit, etc), and then test each vector resulting from varying
each parameter type in turn according to each full range, while fixing
the other parameter types. Once all have been tested, produce a new
nominal vector of parameters picking the best option for each type.
Then start with the new type, and repeat with a new set of tests.
To avoid repeating any test, all performed tests and their results
are saved in a list that is consulted before issuing a new test.
Once a new vector has been found by iterating this procedure a few
times, the process is again restarted, now by selecting the vector
with the lowest MAD among all tests so far. This restart usually
converges very quickly, because all tests neigbouring to the optimal
have usually already been performed.

Depennding on how close to the minimum the algorithm is started an
the ranges allowed, this procedure usually cuts down the number of
required tests by a factor of 5-20 compared to an exhastive search.

"""

import copy
import numpy as np


class TestParams:
    """Data class that defines the parameters to be varied. There is an equivalency 
    defined that excludes the mad value.
    """
    def __init__(self, klip, fitrad, bBG, bDark, bStat):
        self.klip = klip
        self.fitrad = fitrad
        self.bBG = bBG
        self.bDark = bDark
        self.bStat = bStat
        self.mad = None
        
    def __eq__(self, other):
        return (self.klip == other.klip and
                self.fitrad == other.fitrad and
                self.bBG == other.bBG and
                self.bDark == other.bDark and
                self.bStat == other.bStat)

    def __str__(self):
        if self.mad is None:
            return 'k: {:d}, r: {:d}, bg: {:d}, d: {:d}, s: {:d}'.format(
                self.klip, self.fitrad, self.bBG, self.bDark, self.bStat)
        return 'k: {:d}, r: {:d}, bg: {:d}, d: {:d}, s: {:d}, mad = {:.2f}'.format(
            self.klip, self.fitrad, self.bBG, self.bDark, self.bStat, self.mad)
    
    
class FindOptimal:
    def __init__(self, klips, fitrads, bBG, bDark, bStat):
        # Parameters to test
        self.klips = np.array(klips, int)      # Number of principal components in PSF fit
        self.fitrads = np.array(fitrads, int)  # Radii of domain over to fit PSF
        self.bBG = bBG          # Test fit background jointly with PSF?
        self.bDark = bDark      # Test subtract dark?
        self.bStat = bStat      # Test subtract static residuals?
        
        self.tested_params = []
        self.current_tests = []
        self.current_test_id = 0
        self.num_tests = 0
        self.iter = 0


    def get_mad(self, tp):
        for t in self.tested_params:
            if tp == t:
                return t.mad
        return None


    def str_tested(self):
        """Return string with parameters of all tests, sorted 
        inversely according to mad
        """
        str_list = np.array([str(test) for test in self.tested_params])
        mad_list = [test.mad for test in self.tested_params]
        ind = np.flip(np.argsort(mad_list))
        
        ret_str = ''
        for t in str_list[ind]:
            ret_str += f'\n{t}'
        return ret_str


    def start(self, nominal_params, max_iter=5):
        self.iter = max_iter
        self.generate_test_series(nominal_params)


    def get_best(self):
        mads = [t.mad for t in self.tested_params]
        return copy.copy(self.tested_params[np.argmin(mads)])


    def next_test(self, mad=None):
        if mad is not None:
            self.current_tests[self.current_test_id].mad = mad
            self.tested_params.append(self.current_tests[self.current_test_id])
            self.current_test_id += 1

        if self.current_test_id == self.num_tests:
            new_params = copy.copy(self.nominal_params)
            nominal_mad = self.get_mad(self.nominal_params)

            ik = len(self.klips)
            ifr = len(self.fitrads)
            ind0 = 0
            klip_mad = [t.mad for t in self.current_tests[ind0:(ind0+ik)]]
            ind0 += ik
            new_params.klip = self.klips[np.argmin(klip_mad)]
            fitrad_mad = [t.mad for t in self.current_tests[ind0:(ind0+ifr)]]
            ind0 += ifr
            new_params.fitrad = self.fitrads[np.argmin(fitrad_mad)]
            if self.bBG:
                new_params.bBG = (self.current_tests[ind0].bBG !=
                      (self.current_tests[ind0].mad > nominal_mad))
                ind0 += 1
            if self.bDark:
                new_params.bDark = (self.current_tests[ind0].bDark !=
                        (self.current_tests[ind0].mad > nominal_mad))
                ind0 += 1
            if self.bStat:
                new_params.bStat = (self.current_tests[ind0].bStat !=
                        (self.current_tests[ind0].mad > nominal_mad))
            self.iter -= 1
            if self.iter > 0:
                self.generate_test_series(new_params)
                return self.next_test()
            return None     # Finished with tests
        mad = self.get_mad(self.current_tests[self.current_test_id])
        if mad is not None:
            self.current_tests[self.current_test_id].mad = mad
            self.current_test_id += 1
            return self.next_test()
        
        return copy.copy(self.current_tests[self.current_test_id])

        
    def generate_test_series(self, test_params):
        self.current_test_id = 0
        self.current_tests = []
        self.nominal_params = copy.copy(test_params)
        
        for klip in self.klips:
            new_test = copy.copy(test_params)
            new_test.klip = klip
            self.current_tests.append(new_test)
            
        for fitrad in self.fitrads:
            new_test = copy.copy(test_params)
            new_test.fitrad = fitrad
            self.current_tests.append(new_test)
        
        if self.bBG:
            new_test = copy.copy(test_params)
            new_test.bBG = not test_params.bBG
            self.current_tests.append(new_test)
        
        if self.bDark:
            new_test = copy.copy(test_params)
            new_test.bDark = not test_params.bDark
            self.current_tests.append(new_test)

        if self.bStat:
            new_test = copy.copy(test_params)
            new_test.bStat = not test_params.bStat
            self.current_tests.append(new_test)
    
        self.num_tests = len(self.current_tests)
    
    
if __name__ == '__main__':
    
    # Simple tests
    
    fo = FindOptimal(klips=[1,3,5,10], fitrads=[25,30,40,45,50,60], bBG=True, bDark=True, bStat=True)
    nominal = TestParams(klip=1, fitrad=30, bBG=True, bDark=True, bStat=True)
    
    for n in range(3):
        fo.start(nominal, 5)
        mad = None
        while True:
            testparams = fo.next_test(mad)
            if testparams is None:
                break
            mad = 100*np.random.rand()
            print('Testing: {:s}, imad = {:.2f}'.format(str(testparams), mad))
        nominal = fo.get_best()    
        print('--- Iter best {:d}: {:s}'.format(n, str(nominal)))
    print('Done!')
    
    