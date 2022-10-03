# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:33:43 2022

@author: JK-WORK
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize("CUSUM_opt.pyx",compiler_directives={"language_level": "3"}),include_dirs=[np.get_include()]) #for the one without annotations

#python CUSUM_opt_setup.py build_ext --inplace