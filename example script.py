#!/usr/bin/env python3
"""
Author: Joshua Stuckner
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from smart_tem.data_class import *
from smart_tem import operations
from smart_tem import inout
from smart_tem import visualize

# Load file
folder = r'E:\E_Documents\Research\SMART-Imaging\Full Datasets//'
folder = r'E:\E_Documents\Research\SMART-Imaging\Full Datasets\OneView\CycloDextrine\Josh//'
file_name = 'Noisy'
file_name = 'OneView1'
fpath = ''.join((folder,file_name,'.hdf5'))
data = ImageSet(fpath)

# Downsample according to the Nyquist theorum.
data.frames = operations.down_sample(data.frames,0.0105, 3)

# Perform chambolle denoising.
b = operations.tv_chambolle(data.frames, 0.1, 0.0002, max_stacking=1)

# Play the movie for visualization.
visualize.play_movie(b, fps=5)
