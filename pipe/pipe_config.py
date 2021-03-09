# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:43:28 2021

@author: Alexis Brandeker, alexis@astro.su.se
"""

import os
ref_lib_path = os.path.join(
    os.path.dirname(__file__), os.path.pardir, 'ref_lib_data'
)
data_root = os.path.join(
    os.path.dirname(__file__), os.path.pardir, 'data_root'
)

if not os.path.exists(ref_lib_path):
    print("No reference data found.")

if not os.path.exists(data_root):
    os.makedirs(data_root)