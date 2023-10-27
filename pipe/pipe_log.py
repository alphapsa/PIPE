# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 21:45:10 2021

@author: Alexis Brandeker, alexis@astro.su.se

Contains the class PipeLog that handles loging activities to
file and stdout for diagnostics.
"""
import time


class PipeLog:
    """ Keeps track of tima and messages to the logfile
    """
    def __init__(self, logfile=None, verbosity=1):
        """If logfile exists, entries will be appended. verbosity
        is an integer that determines what messages should be printed
        on screen (all messages go to the log). A higher verbosity
        means more messages will be printed. Default is verbosity=1.
        """
        self.logfile = logfile
        self.verbosity = verbosity
        self.init_time()
        
    def init_time(self):
        """ The initial time is referred to when computing
        time passed, e.g. to find execution time so far.
        """
        self.t0 = time.time()
        
    def mess(self, text, level=1):
        """ Saves message to logfile and prints on screen if 
        integer level is equal to verbosity or lower.
        """
        str = ('{:s} [{:.2f} min] {:s}'.format(time.asctime(), 
              (time.time()-self.t0)/60, text))
        if level <= self.verbosity:
            print(str)        
        # Always save to logfile. Catch exception if operation fails.
        # (to avoid crash due to e.g. busy file system)
        try:
            with open(self.logfile, 'a') as logfile:
                logfile.write(str+'\n')
        except:
            print('Warning: could not save log to {:s}'.format(self.logfile))

    def mess_list(self, text_list, level=1):
        """ Walks thorugh list of strings, saves them to logfile and
        prints on screen if integer level is equal to verbosity or lower.
        """
        str = ('{:s} [{:.2f} min] ---------'.format(time.asctime(), 
              (time.time()-self.t0)/60))
        if level <= self.verbosity:
            print(str)        
        # Always save to logfile. Catch exception if operation fails.
        # (to avoid crash due to e.g. busy file system)
        try:
            with open(self.logfile, 'a') as logfile:
                logfile.write(str+'\n')
                for text in text_list:
                    logfile.write(text+'\n')
                    if level <= self.verbosity:
                        print(text)
                logfile.write('------------------\n')
        except:
            print('Warning: could not save log to {:s}'.format(self.logfile))
        if level <= self.verbosity:
            print('------------------')        
        