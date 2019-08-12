#!/usr/bin/env python3

# Author: Joshua Stuckner
# Date: 2017/06/21

import warnings
import time
import glob
import os

import numpy as np
from skimage.util import img_as_ubyte
import scipy.misc
#from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import ndimage
from PIL import Image
import cv2
import h5py


from smart_preprocess.dmreader import digital_micrograph as dm

from smart_preprocess import visualize


def uint8(img):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = img_as_ubyte(img)
    return img


def load(path):
    #ERROR, THIS LOADS IMAGES WITH VALUES FROM 0 TO 254 NOT 255!
    dmtypes = ['.dm3', '.dm4']
    if any(t in path for t in dmtypes):
        return dm_to_npy(path)
    img = scipy.misc.imread(path, mode='I')
    display_max = np.percentile(img, 99.9)
    display_min = np.percentile(img, 0.1)
    img = float32_to_uint8(img, display_min, display_max)
    return img


def float32_to_uint8(image, display_min, display_max):
    image = np.array(image, copy=True)
    image = image.clip(display_min, display_max, out=image)
##    plt.imshow(image, cmap=plt.cm.gray)
##    plt.show()
    image = np.add(image, -display_min, casting='unsafe')
    image = np.divide(image, (display_max + 1 - display_min) / 256, casting='unsafe')

##    plt.imshow(image, cmap=plt.cm.gray)
##    plt.show()
    image = image.astype(np.uint8)
##    plt.imshow(image, cmap=plt.cm.gray)
##    plt.show()
    return image

def uint16_to_uint8(image, display_min, display_max):
    image = np.array(image, copy=True)
    image = image.clip(display_min, display_max, out=image)
    #image -= display_min
    image = np.add(image, -display_min, casting='unsafe')
    #image //= (display_min - display_max + 1) / 256.
    image = np.divide(image, (display_max + 1 - display_min) / 256, casting='unsafe')
    image = image.astype(np.uint8)
    return image

# def save_image(im, save_name, normalize=False):
#     std = im.std()
#     vmin = im.min() + std / 3
#     vmax = im.max() - std / 3
#     if normalize:
#         imsave(save_name, )
#         ims.append([plt.imshow(frames[:, :, i], vmin=vmin, vmax=vmax,
#                                cmap=plt.cm.gray, animated=True)])
#     else:
#         ims.append([plt.imshow(frames[:, :, i], vmin=0, vmax=255,
#                                cmap=plt.cm.gray, animated=True)])

def save_movie(frames, save_name, fps=20, bit_rate=-1, normalize=False,
               sharpen=False):
    """
    Saves a movie of the images in frames.

    Parameters
    ----------
    frames : ndarray (3D)
        Contains images to create a movie from.
    fps : double, optional
        Frames per second of movie.  Defaults is 20.
    save_name : string
        Path to save the movie.
    bit_rate : int
        bits per second of movie.  See matplotlib.animation.writers
    """

    print('Saving movie...', end=' ')
    start = time.time()
    fig = plt.figure(dpi=100)
    fig.patch.set_facecolor('black')
    plt.axis('off')

    plt.gca().set_position([0, 0, 1, 1])
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=None, hspace=None)
    ims = []
    
    try:
        rows, cols, num_frames = frames.shape
    except AttributeError:
        list_frames = frames.copy()
        num_frames = len(list_frames)
        rows, cols = list_frames[0].shape
        frames = np.zeros((rows,cols,num_frames))
        for i, frame in enumerate(num_frames):
            frames[:,:,i] = frame

    fig.set_size_inches(cols/100, rows/100)
    
##    writer = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'FMP4'), bit_rate,
##                             (cols, rows), False)
##    for i in range(num_frames):
##        writer.write(frames[:,:,i])
##
##    writer.release()
    
    #For normalizing
    if normalize:
        std = frames.std()
        vmin = frames.min()+std/2
        vmax = frames.max()-std/2

##    FFMpegWriter = animation.writers['ffmpeg']
##    writer = FFMpegWriter(fps=fps, bitrate=bit_rate, extra_args=['-vcodec', 'libx264',
##                                                     '-pix_fmt', 'yuv420p'])
##    fig = plt.figure()
##    ax = plt.subplot(111)
##    plt.axis('off')
##    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
##    ax.set_frame_on(False)
##    ax.set_xticks([])
##    ax.set_yticks([])
##    plt.axis('off')
##
##    im = ax.imshow(frames[:,:,0],interpolation='nearest')
##    with writer.saving(fig, save_name, dpi=10):
##        for i in range(num_frames):
##            ax.imshow(frames[:,:,i],interpolation='nearest')
##            writer.grab_frame()   
    
    for i in range(num_frames):
        #visualize.update_progress(i / num_frames)
        if sharpen:
            sigma = 1
            alpha = 0.5
            blurred = ndimage.gaussian_filter(frames[:,:,i], sigma)
            filter_blurred = ndimage.gaussian_filter(blurred, 1)
            frames[:,:,i] = blurred + alpha * (blurred - filter_blurred)
        if normalize:
            ims.append([plt.imshow(frames[:,:,i], vmin=vmin, vmax=vmax,
                                   cmap=plt.cm.gray, animated=True)])
        else:
            ims.append([plt.imshow(frames[:,:,i], vmin=0, vmax=255,
                                   cmap=plt.cm.gray, animated=True)])

    ani = animation.ArtistAnimation(fig, ims, fps / 1000, True, fps / 1000)
    writer = animation.writers['ffmpeg'](fps=fps, bitrate=bit_rate,
                         extra_args=['-vcodec', 'libx264',
                                     '-pix_fmt', 'yuv420p'])
##    writer = animation.writers['ffmpeg'](fps=fps, bitrate=bit_rate,
##                                         extra_args=['-vcodec', 'libx264',
##                                                     '-s', '1280x960', #sets frame size
##                                                     '-pix_fmt', 'yuv420p'])
    ani.save(save_name, writer=writer)
    print('Done, took', round(time.time() - start, 2), 'seconds.')

def save_images(frames, folder, normalize=False):
    _,_, num_frames = frames.shape
    if folder[:-1] != '/':
        folder = ''.join((folder,'/'))
    
    for i in range(num_frames):
        if i < 10:
            space = '00'
        elif i < 100:
            space = '0'
        else:
            space = ''
        im_path = ''.join((folder, space,str(i),".tif"))
        scipy.misc.imsave(im_path,frames[:,:,i])

def get_file_names(folder, ftype=None):
    if folder[-1] != '/':
        folder = ''.join((folder, '/'))
    if ftype is not None and ftype[0] == '.':
        ftype = ftype[1:]                        
    if ftype is None:
        files = os.listdir(folder)
        out = []
        for file in files:
            out.append(''.join((folder,file)))
        return out
    else:
        return glob.glob(''.join((folder, '*.', ftype)))
    
def files_in_subfolders(folder, ftype):
    return [file for file in glob.glob(''.join((folder + '/**/*.', ftype)), recursive=True)]

def rename_image_clipper(folder):
    files = os.listdir(folder)
    for file in files:
        os.rename(''.join((folder,file)),
                  ''.join((folder, file.split('.')[0], '.png')))

def create_folder(folder):
    # Checks to see if the directory exists and creates it if it doesn't.
    try:
        os.stat(folder)
    except:
        os.mkdir(folder)
    
def frames_from_hdf5(file):
    """
    Returns the space-time cube array of the movie data in an hdf5 file.
    """
    # Load all images
    print('Loading frames...', end=' ')
    start = time.time()
    f = h5py.File(file, "r")
    frames = f['data'][()]
    print('Done, took', round(time.time() - start, 2), 'seconds.')
    return frames

def dm_to_npy(dm_path):
    """
    Converts a dm3 or dm4 image to a uint8 numpy array with a contrast limit
    suggested in the dm file.
    """
    # Load the dm4 file and extract the image data.
    signal = dm.file_reader(dm_path)[0]
    image_data = signal['data']
    # get the contrast limits from the file
    max_trim_percent = signal['original_metadata']['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['EstimatedMaxTrimPercentage']
    min_trim_percent = signal['original_metadata']['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['EstimatedMinTrimPercentage']
    display_max = np.percentile(image_data, 100-min_trim_percent*100)
    display_min = np.percentile(image_data, min_trim_percent*100)
    # convert to uint8
    if image_data.dtype == np.uint16:
        image_data = uint16_to_uint8(
            image_data, display_min, display_max)
    elif image_data.dtype == np.float32:
        image_data = float32_to_uint8(
            image_data, display_min, display_max)
    return image_data

def dm_to_tiff(input_folder, output_folder, convert_sub_folders=False,
               subset=None):
    """
    Converts dm4 files to tiff format.
    convert_sub_folder : converts files in all subfolders reccursively.
    subset : (a,b) only converts files between a and b (sorted alphabetically)
    """

    print('Converting Files...', end=' ')
    start = time.time()
    
    
        
    # Create output folder.
    create_folder(output_folder)

    # Get list of dm files.
    if convert_sub_folders:
        file_paths = glob.glob(''.join((input_folder, '/**/*.dm*')),
                               recursive=True)
    else:
        file_paths = glob.glob(''.join((input_folder, '/*.dm*')))

    if subset is not None:
        file_paths = file_paths[subset[0]:subset[1]]

    for i, path in enumerate(file_paths):
        #signal = hs.load(path)
        signal = dm.file_reader(path)[0]
        image_data = signal['data']

        # get the contrast limits from the file

        max_trim_percent = signal['original_metadata']['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['EstimatedMaxTrimPercentage']
        min_trim_percent = signal['original_metadata']['DocumentObjectList']['TagGroup0']['ImageDisplayInfo']['EstimatedMinTrimPercentage']
        display_max = np.percentile(image_data, 100-min_trim_percent*100)
        display_min = np.percentile(image_data, min_trim_percent*100)

        if i == 0:
            print('\nimage, rows, cols, unit, pixel size')

        if image_data.dtype == np.uint16:
            image_data = uint16_to_uint8(
                image_data, display_min, display_max)
        elif image_data.dtype == np.float32 or image_data.dtype == np.int32:
            image_data = float32_to_uint8(
                image_data, display_min, display_max)
            
        fname = path.split('\\')[-1]
        fname = fname.split('.dm')[0]
        print(fname,
              signal['axes'][0]['size'],
              signal['axes'][1]['size'],
              signal['axes'][0]['units'],
              signal['axes'][0]['scale'], sep=', ')
        output_path = ''.join((output_folder, '/', fname, '.tif'))
        #signal.save(output_path)
        #scipy.misc.imsave(output_path, image_data)

        scipy.misc.toimage(image_data, cmin=0, cmax=255).save(output_path)



    print('Done, took', round(time.time() - start, 2), 'seconds.')
                                
