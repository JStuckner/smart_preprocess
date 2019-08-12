#!/usr/bin/env python3
"""
This module applies global filters to the image sets

Author: Joshua Stuckner
Date: 2017/06/21
"""

import warnings

import sys
import time
import math

import warnings
from scipy.ndimage import filters
from scipy.misc import imresize, imsave
import scipy.ndimage as ndim
from skimage import restoration, morphology
from skimage.filters import threshold_otsu
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff

from stuckpy.microscopy import inout
from stuckpy.microscopy import visualize


def polynomial_fit_normalize(im, mask=None, return_fit=False, dtype='float32',
                             fit_path=None):
    """
    Fits a 3D surface to the inensity of pixels using 2nd order poly fit
    and subtracts the fit from the image to remove trends.
    """
    from astropy.modeling import models, fitting
    rows, cols = im.shape
    y, x = np.mgrid[:rows,:cols]

    if mask is not None:
        m = np.ma.masked_array(im, mask=mask)
    else:
        m = im

    # Fit the data using astropy
    p_init = models.Polynomial2D(degree=2)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, m)
    fit = p(x,y)

    if fit_path is not None:
        imsave(fit_path, fit.astype('uint8'))

    # subtract fit from image
    
    fit_minus_mean = fit - np.mean(fit)


    
    if dtype == 'uint8':
        im = np.clip(np.subtract(im, fit_minus_mean), 0, 255).astype(dtype)
    else:
        im = np.subtract(im, fit_minus_mean).astype(dtype)
    

    if return_fit:
        return im, fit
    else:
        return im

# update_progress() : Displays or updates a console progress bar
# Accepts a float between 0 and 1. Any int will be converted to a float.
# A value under 0 represents a 'halt'.
# A value at 1 or bigger represents 100%
def update_progress(progress):
    barlength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barlength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format(
        "#"*block + "-"*(barlength-block), round(progress*100, 1), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def gaussian_stacking(image_set, sigma=1):
    """
    Stacks the images using a Gaussian filter

    This function is a wrapper around scipy.ndimage.filters.gaussian_filter
    """
    print('Applying Guassian stacking...', end=' ')
    start = time.time()
    out = filters.gaussian_filter(image_set, (0, 0, sigma))
    print('Done, took', round(time.time()-start, 2), 'seconds.')
    return out

def gaussian(image_set, sigma=None):
    """
    Applies a gaussian filter to the image set.

    This function is a wrapper around scipy.ndimage.filters.gaussian_filter
    """
    if sigma is None:
        print('Please set a filter radius (sigma).  Defaulting to sigma = 1.')
        sigma = 1

    #print('Applying guassian filter...', end=' ')
    #start = time.time()
    try:
        out = filters.gaussian_filter(image_set, (sigma, sigma, 0))
    except RuntimeError: # When there is only one frame
        out = filters.gaussian_filter(image_set, (sigma, sigma))
    #print('Done, took', round(time.time()-start, 2), 'seconds.')
    return out


def median_stacking(image_set, size=3):
    print('Applying median stacking...', end=' ')
    start = time.time()
    out = filters.median_filter(image_set, size=(1,1,size))
    print('Done, took', round(time.time() - start, 2), 'seconds.')
    return out

def median(image_set, size=None):
    """
    Applies a median filter to the image set.

    This function is a wrapper around scipy.ndimage.filters.median_filter
    """
    if size is None:
        print('Please set a filter size.  Defaulting to size = 3.')
        size = 3

    print('Applying median filter...', end=' ')
    start = time.time()

    try:
        out = filters.median_filter(image_set, size=(size, size, 1))
    except RuntimeError: # When there is only one frame
        out = filters.median_filter(image_set, size=(size, size))

    print('Done, took', round(time.time() - start, 2), 'seconds.')
    return out

def down_sample(image_set, pixel_size, nyquist=3, d=0.1, source_sigma=0.5,
                dtype='uint8'):
    """
    This function performs Gaussian downsampling on the frames.  The goal is
    to resample the image to a pixel size that is between 2.3 and 3 times
    smaller than the point to point resolution of the microscope. This is
    in accordance with the Nyquist-Shannon sampling theorem.  The optimum
    value of the Gaussian filter sigma is calculated.

    :param image_set:
        3D ndarray.  Contains the image frames.
    :param pixel_size:
        The size of the pixels in nanometers.
    :param d: optional
        The point to point resolution of the microscope.
    :param nyquist: optional
        Should be between 2.3 - 3.
        Sets the number of pixels per d
    :param source_sigma: optional
        Set higher than 0.5 if the image was already blurred and a sharper
        downsampled result is desired.

    :return:
        The downsampled dataset
    """
    try:
        rows, cols, num_frames = image_set.shape
    except ValueError:
        rows, cols = image_set.shape
        num_frames = 1

    target_pixel_size = d / nyquist
    target_sigma = 0.5
    scale = pixel_size / target_pixel_size

    if pixel_size >= target_pixel_size:
        print('Down sampling will not reach the target (need interpolation).')
        return image_set
    else:
        # Calculate the optimum Gaussian filter sigma.
        s = target_sigma * target_pixel_size / pixel_size
        gsigma = math.sqrt(s**2 - source_sigma**2)
        # Apply the gaussian filter.
        out = gaussian(image_set, gsigma)
        print('Downsampling...', end=' ')
        start = time.time()
        if num_frames == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = ndim.interpolation.zoom(out, (scale,scale),
                                              order = 3, prefilter=True,
                                              output=dtype)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = ndim.interpolation.zoom(out, (scale, scale, 1),
                                              order= 3, prefilter=True,
                                              output=dtype)
        print('Done, took', round(time.time() - start, 2), 'seconds.')
        return out


def tv_chambolle(image_set, weight, max_stacking=None, eps=0.0002, verbose=True):
    """Performs chambolle filtering using the skimage library.
    :param image_set:
    :param weight:
    :param eps:
    :param max_stacking:
    :return:
    """
    if verbose:
        print('Applying Total Variation Chambolle filter...', end=' ', flush=True)
    start = time.time()

    try:
        rows, cols, num_frames = image_set.shape
    except ValueError:
        rows, cols = image_set.shape
        num_frames = 1

    if num_frames == 1:
        if verbose:
            print("Performing filter on the single frame passed...", end=" ")
        out = restoration.denoise_tv_chambolle(
            image_set, weight=weight, eps=eps)
    elif max_stacking is None or max_stacking < 0:
        out = inout.uint8(restoration.denoise_tv_chambolle(
            image_set, weight=weight, eps=eps))
    elif not isinstance(max_stacking, int):
        print("Max stacking must be an odd integer")
        return image_set
    elif num_frames <= max_stacking:
        print(("Total number of frames is <= max_stacking. "
               "Stacking not limited."))
        out = inout.uint8(restoration.denoise_tv_chambolle(
            image_set, weight=weight, eps=eps))
    elif max_stacking < 1:
        print("max_stacking must be greater than 0.")
        return image_set
    elif max_stacking % 2 == 0:
        print("Max stacking must be an odd integer")
        return image_set
    else:
        out = np.zeros((rows, cols, num_frames))
        if max_stacking == 1:
            for i in range(num_frames):
                if verbose:
                    if i % 10 == 0:
                        visualize.update_progress(i / num_frames)
                out[:, :, i] = inout.uint8(restoration.denoise_tv_chambolle(
                    image_set[:, :, i],
                    weight=weight, eps=eps))
        else:
            half_max = int(max_stacking - 1 / 2)
            for i in range(num_frames):
                if verbose:
                    if i % 10 == 0:
                        visualize.update_progress(i / num_frames)
                if i < max_stacking:
                    out[:,:,i] = inout.uint8(
                        restoration.denoise_tv_chambolle(
                        image_set[:,:,:i+half_stack],
                        weight=weight, eps=eps))[:,:,i]
                elif num_frames - i > num_frames - max_stacking:
                    out[:, :, i] = inout.uint8(
                        restoration.denoise_tv_chambolle(
                        image_set[:,:,i-half_stack:],
                        weight=weight, eps=eps))[:,:,-1*(num_frames-i)]
                else:
                    out[:, :, i] = inout.uint8(
                        restoration.denoise_tv_chambolle(
                        image_set[:,:,i-half_stack:i+half_stack],
                        weight=weight, eps=eps))[:,:,half_stack]

    if verbose:
        print('Done, took', round(time.time() - start, 2), 'seconds.')
    return out.astype('uint8')

def tv_bregman(image_set, weight, eps, max_iter=100, isotropic=True):
    """Performs Bregman denoising using the skimage library.
    :param image_set:
    :param weight:
    :param eps:
    :param max_stacking:
    :return:
    """
    print('Applying Total Variation Bregman...', end=' ')
    start = time.time()
    out = inout.uint8(restoration.denoise_tv_bregman(
        image_set, weight=weight, eps=eps, isotropic=isotropic,
        max_iter=max_iter))

    print('Done, took', round(time.time() - start, 2), 'seconds.')
    return out.astype('uint8')

def normalize(im, std=-1, verbose=True):
    im = im.astype('float')
    if std < 0:
        minval = im.min()
        maxval = im.max()
    else:
        shift = std * np.std(im)
        minval = int(round(np.average(im) - shift,0))
        maxval = int(round(np.average(im) + shift,0))
        
    if minval != maxval:
        im -= minval
        im *= (255.0/(maxval-minval))
        im = np.clip(im, 0.0, 255.0)
        im = im.astype('uint8')

    if verbose:
        print('Levels balanced.')

    return im
    
def get_background_mask(image_set):
    stacked = np.average(image_set, axis=2) # Stack all images
    time_SD = np.std(image_set, axis=2) #variation of each pixel through time
    std_mult = 1 #adjust threshold value
    thresh = time_SD > threshold_otsu(time_SD) + std_mult * np.std(time_SD)
    small = 10 #size of small objects to remove for first round
    removed = morphology.remove_small_objects(thresh, small)
    selem_radius = 3 #dialtion radius
    dial = morphology.binary_dilation(
        removed, selem=morphology.disk(selem_radius))
    small2 = 300 #size of small objects to remove for second round
    removed2 = morphology.remove_small_objects(dial, small2)
    fr = 30 #final closing radius
    pad = np.pad(removed2, fr, mode='constant') #prevents over erosion near edge         
    final_mask = morphology.binary_closing(
        pad, selem=morphology.disk(fr))[fr:-fr, fr:-fr]
    area = np.count_nonzero(final_mask)
    rad = math.sqrt(area/3.14)
    fdr = int(round(rad/5)) #final dilation radius
    print(fdr)
    final_mask = morphology.binary_dilation(
        final_mask, selem=morphology.disk(fdr))
    mask3D = np.zeros(image_set.shape, dtype=bool)
    mask3D[:,:,:] = final_mask[:,:, np.newaxis]
    #image_set[mask3D == False] = set_to
    
##    ims = [stacked,
##           time_SD,
##           thresh,
##           removed,
##           dial,
##           removed2,
##           final_mask]
##    titles = ["Stacked",
##              "Time St. Dev.",
##              "Thresholded",
##              "Remove small",
##              "Dialated",
##              "Remove 2",
##              "Final mask"]

    #visualize.plot_grid(ims,titles, rows=2, cols=4)
    #visualize.play_movie(image_set)

    return mask3D

    
def unsharp_mask(im, sigma, threshold, amount):
    print('Applying Unsharp Mask...', end=' ')
    start = time.time()
    try:
        rows, cols, num_frames = im.shape
    except ValueError:
        rows, cols = im.shape
        num_frames = 1

    if amount > 0.9 or amount < 0.1:
        print("'amount' should be between 0.1 and 0.9!")
        
    if num_frames == 1:
        blurred = filters.gaussian_filter(im, (sigma, sigma))
        lowContrastMask = abs(im - blurred) < threshold
        sharpened = im*(1+amount) + blurred*(-amount)
        locs = np.where(lowContrastMask != 0)
        out = im.copy()
        out[locs[0], locs[1]] = np.clip(sharpened[locs[0], locs[1]], 0, 255)
        print('Done, took', round(time.time() - start, 2), 'seconds.')
        return out

    else:
        out = im.copy()
        blurred = filters.gaussian_filter(im, (sigma, sigma, 0))
        for i in range(num_frames):
            lowContrastMask = abs(im[:,:,i] - blurred[:,:,i]) < threshold
            sharpened = im[:,:,i]*(1+amount) + blurred[:,:,i]*(-amount)
            locs = np.where(lowContrastMask != 0)
            out[:,:,i][locs[0], locs[1]] = np.clip(sharpened[locs[0], locs[1]], 0, 255)
        print('Done, took', round(time.time() - start, 2), 'seconds.')
        return out

def blur_background(im, sigma=None, thresh=None,
                    small1=None, dil_rad=None,
                    small2=None, close_rad=None):

    rows, cols, _ = im.shape
    size = (rows + cols) / 2
    
    # get standard deviation
    std = np.std(im, axis=2)

    # threshold
    if thresh is None:
        thresh = int(threshold_otsu(std) * 0.75)
    binary = std > thresh

    # set values if none
    sigma = size/100 if sigma == None else sigma
    small1 = int(round(size/10)) if small1 == None else small1
    dil_rad = int(round(size/100)) if dil_rad == None else dil_rad
    small2 = int(round(size)) if small2 == None else small2
    close_rad = int(round(size/20)) if close_rad == None else close_rad

    #morphology operations
    morph = binary.copy()
    selem_radius = 1
    morph = morphology.remove_small_objects(morph, small1)
    morph = morphology.binary_dilation(
                morph, selem=morphology.disk(dil_rad))
    morph = morphology.remove_small_objects(morph, small2)
    fr = close_rad #final closing radius
    pad = np.pad(morph, fr, mode='constant') #prevents over erosion near edge         
    morph = morphology.binary_closing(
        pad, selem=morphology.disk(fr))[fr:-fr, fr:-fr]

    # make location masks
    mask = np.where(morph != 0)
    notmask = np.where(morph == 0)

    # blur the not mask
    allBlur = gaussian(im, sigma)
    blur = im.copy()
    blur[notmask[0], notmask[1], :] = allBlur[notmask[0], notmask[1], :]

    return blur

def remove_outliers(image_set, percent=0.1, size=3):
    """
    Replaces pixels in the top and bottom percentage of the pixels in each
    image with a median pixel value of the pixels in a window of
    size by size pixels.
    """
    print('Removing outliers...', end=' ')
    start = time.time()

    try:
        rows, cols, num_frames = image_set.shape
    except ValueError:
        rows, cols = image_set.shape
        num_frames = 1

    for i in range(num_frames):
        if i % 10 == 0:
            visualize.update_progress(i / num_frames)
        im = image_set[:,:,i]
        #Create a median filter
        med = filters.median_filter(im, size=size)

        #  Create the outlier mask
        low_outlier = np.percentile(im, percent)
        high_outlier = np.percentile(im, 100-percent)
        mask = np.zeros((rows,cols), dtype='bool')
        mask[im >= high_outlier] = 1
        mask[im <= low_outlier] = 1

        # If there are many outliers together, they are probably not outliers.
        # scale the amount to multiply ther percent.
        if percent > 9:
            pmult = 2
        elif percent > 1:
            pmult = 3
        else:
            pmult = 4

        low_outlier = np.percentile(im, percent * pmult)
        high_outlier = np.percentile(im, 100 - percent * pmult)
        mask2 = np.zeros((rows,cols), dtype='bool')
        mask2[im >= high_outlier] = 1
        mask2[im <= low_outlier] = 1
        mask2 = morphology.remove_small_objects(mask2, 12)
        mask[mask2==1] = 0

        
        im[mask == 1] = med[mask==1]

    print('Done, took', round(time.time() - start, 2), 'seconds.')
    return image_set
        
        

