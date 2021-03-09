# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:43:28 2021

@author: Alexis Brandeker, alexis@astro.su.se
"""
import socket

# This code assigns path names based on what the name of
# the host system is. This is used to be able to run the
# code on different systems without having to remember
# to change the path variables. The paths need to be
# entered the first time the code is run on a system.

hostname = socket.gethostname()
if hostname == 'rubicon':
    ref_lib_path = r'/home/alexis/CloudStation/CHEOPS/Data/REF'
    data_root = r'/AstroData/CHEOPSDATA'
elif hostname == 'Hydra':
    ref_lib_path = r'D:\alexis\CloudStation\CHEOPS\Data\REF'
    data_root = r'L:\CHEOPSDATA'
else:
    print('ERROR: Hostname \"{:s}\" not configured.'.format(hostname))
    exit()
print('{:s} is host'.format(hostname))

if __name__=='__main__':
    print('Do not run')
    