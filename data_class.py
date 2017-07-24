#!/usr/bin/env python3
"""
This module contains the ImageSet class.

Author: Joshua Stuckner
Date: 2017/06/21
"""

import glob
import time

import numpy as np
import h5py

from smart_tem import inout
from smart_tem import operations


class ImageSet:
    def __init__(self, path):
        """
        This class accepts hdf5 files
        """

        # Initiate variables
        self.path = path


        # Load all images
        print('Loading frames...', end=' ')
        start = time.time()
        f = h5py.File(path, "r")
        self.frames = f['data'][()]
        print('Done, took', round(time.time() - start, 2), 'seconds.')

    def apply_filter(self, filter, sigma=None):
        supported = ['gaussian', 'median']

        if type(filter) is not str or filter.lower() not in supported:
            sup = []
            for s in supported:
                sup.append(''.join(('"', s, '"')))
            raise TypeError('filter must be one of the following strings: '
                            + ', '.join(sup) + '.  For more '
                            + 'filters see scipy.ndimage.filters')

        if filter.lower() == 'gaussian':
            self.frames = operations.gaussian(self.frames, sigma=sigma)
        if filter.lower() == 'median':
            self.frames = operations.median(self.frames, sigma=sigma)
