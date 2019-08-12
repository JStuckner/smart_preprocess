#!/usr/bin/env python3

# Author: Joshua Stuckner
# Date: 2017/06/21

import time
import os
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
from PIL import Image, ImageTk
from matplotlib.colors import Normalize
from skimage.color import gray2rgb, rgb2gray
import skimage.measure as measure
import scipy.ndimage.morphology as morphology
from smart_preprocess import inout
import warnings

class Image_Viewer(tk.Toplevel):
    def __init__(self, image_set, normalize=False, main=True):
        if main:
            self.tk = tk.Tk()
        else:
            self.tk = tk.Toplevel()
            
        self.image_set = image_set
        self.normalize = normalize
        self.zoomLoc = (0, 0, 0, 0)  # xmin, ymin, xmax, ymax
        self.tk.configure(background='black')
        self.index = 0
        self.rows, self.cols, self.num_frames = image_set.shape

        self.state = False
        self.tk.bind("<F11>", self.toggle_fullscreen)
        #self.tk.bind("<F4>", self.restart_program)
        self.tk.bind("<Escape>", self.end_fullscreen)
        self.tk.bind("<Button-1>", self.click)
        self.tk.bind('<Left>', self.leftKey)
        self.tk.bind('<Right>', self.rightKey)

        # zoom bind
        self.tk.bind("<MouseWheel>", self.zoom)

        # Get size information to test for resize
        self.oldWidth, self.oldHeight = self.tk.winfo_width(), self.tk.winfo_height()

        # Create Canvas
        self.canvas = tk.Label(self.tk, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.configure(background='black')

        self.tk.bind("<Configure>", self.update)

    def zoom(self, event):
        imw, imh = self.image.size
        size = max(self.image.size)
        zoom = int(size / 20)
        xrat = event.x / self.height
        yrat = event.y / self.width

        # Zoom out.
        if event.num == 5 or event.delta == -120:
            xmin = self.zoomLoc[0] - zoom * xrat
            xmax = self.zoomLoc[2] + zoom * (1 - xrat)
            ymin = self.zoomLoc[1] - zoom * yrat
            ymax = self.zoomLoc[3] + zoom * (1 - yrat)
            if ymin >= 0 and xmin >= 0 and ymax <= imw and xmax <= imh:
                self.zoomLoc = (xmin, ymin, xmax, ymax)

        # Zoom in.
        if event.num == 4 or event.delta == 120:
            xmin = self.zoomLoc[0] + zoom * xrat
            xmax = self.zoomLoc[2] - zoom * (1 - xrat)
            ymin = self.zoomLoc[1] + zoom * yrat
            ymax = self.zoomLoc[3] - zoom * (1 - yrat)
            if ymin < ymax and xmin < xmax:
                self.zoomLoc = (xmin, ymin, xmax, ymax)

        self.draw()

    def leftKey(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = self.num_frames - 1
        self.oldWidth = 0  # Cheat to tell update to switch the image
        self.zoomLoc = (0, 0, 0, 0)  # Reset zoom.
        self.update()

    def rightKey(self, event):
        self.index += 1
        if self.index == self.num_frames:
            self.index = 0
        self.oldWidth = 0  # Cheat to tell update to switch the image
        self.zoomLoc = (0, 0, 0, 0)  # Reset zoom.
        self.update()

    def click(self, event):
        if event.x > self.tk.winfo_width() / 2:
            self.index += 1
            if self.index == self.num_frames:
                self.index = 0
        else:
            self.index -= 1
            if self.index < 0:
                self.index = self.num_frames - 1
        self.oldWidth = 0  # Cheat to tell update to switch the image
        self.zoomLoc = (0, 0, 0, 0)  # Reset zoom.
        self.update()

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.tk.attributes("-fullscreen", self.state)
        self.update()
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.tk.attributes("-fullscreen", False)
        self.update()
        return "break"

    def update(self, event=None):

        self.width = self.tk.winfo_width()
        self.height = self.tk.winfo_height()

        # if update is called by an image resize
        if self.oldWidth != self.tk.winfo_width() or self.oldHeight != self.tk.winfo_height():
            self.oldWidth = self.tk.winfo_width()
            self.oldHeight = self.tk.winfo_height()

            self.width, self.height = self.tk.winfo_width(), self.tk.winfo_height()

            w, h = self.width, self.height

            im = self.image_set[:,:,self.index]

            if self.normalize:
                im = im.astype('float')
                minval = self.image_set.astype('float').min()
                maxval = self.image_set.astype('float').max()
                if minval != maxval:
                    im -= minval
                    im *= (255.0/(maxval-minval))
                    im = im.astype('uint8')

            self.image = Image.fromarray(im, mode='L')




        if self.zoomLoc == (0, 0, 0, 0):
            imw, imh = self.image.size
            self.zoomLoc = (0, 0, imw, imh)

        self.draw()
        # self.canvas.update_idletasks()

    def draw(self):
        imw, imh = self.image.size
        imRatio = imw / imh
        camRatio = self.width / self.height

        # print(imRatio - camRatio)

        if imRatio - camRatio > 0.001:
            w = self.width
            h = int(w / imRatio)
        elif imRatio - camRatio < -0.001:
            h = self.height
            w = int(h * imRatio)

        # w, h, = self.width, self.height
        image = self.image.copy()
        image = image.crop(self.zoomLoc)
        #image = image.resize((w, h), Image.ANTIALIAS)
        image = image.resize((w, h))
        self.photo = ImageTk.PhotoImage(image)

        self.canvas.configure(image=self.photo)
        self.canvas.image = image


def showFull(img, title=None, cmap=None, interpolation='none'):
    """
    Displays a full screen figure of the image.

    Parameters
    ----------
    img : ndarray
        Image to display.
    title : str, optional
        Text to be displayed above the image.
    cmap : Colormap, optional
        Colormap that is compatible with matplotlib.pyplot
    interpolation : string, optional
        How display pixels that lie between the image pixels will be handled.
        Acceptable values are ‘none’, ‘nearest’, ‘bilinear’, ‘bicubic’,
        ‘spline16’, ‘spline36’, ‘hanning’, ‘hamming’, ‘hermite’, ‘kaiser’,
        ‘quadric’, ‘catrom’, ‘gaussian’, ‘bessel’, ‘mitchell’, ‘sinc’, ‘lanczos’
    """

    # Show grayscale if cmap not set and image is not color.
    if cmap is None and img.ndim == 2:
        cmap = plt.cm.gray
        
    plt.imshow(img, cmap = cmap, interpolation=interpolation)
    plt.axis('off')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()   
    if title is None:
        plt.gca().set_position([0, 0, 1, 1])
    else:
        plt.gca().set_position([0, 0, 1, 0.95])
        plt.title(title)
    plt.show()
    
def showSkel(skeleton, mask, dialate=False, title=None, returnSkel=False,
             cmap=plt.cm.nipy_spectral, notMask = True):
    """
    Displays skelatal data on top of an outline of a binary mask. For example,
    displays a medial axis transform over an outline of segmented ligaments.

    Parameters
    ----------
    skeleton : 2D array
        Data to be displayed.
    mask : binary 2D array
        Mask of segmentation data, the outline of which is displayed along with
        the skel data.
    dialate : boolean, optional
        If dialate is true, the skelatal data will be made thicker in the
        display.
    title : str, optional
        Text to be displayed above the image.
    """

    skel = np.copy(skeleton)

    # Find the outlines of the mask and make an outline mask called outlines.
    contours = measure.find_contours(mask, 0.5)
    outlines = np.zeros((mask.shape), dtype='uint8')
    for n, contour in enumerate(contours):
        for i in range(len(contour)):
            outlines[int(contour[i,0]), int(contour[i,1])] = 255

    # Make the skel data thicker if dialate is true.
    if dialate:
        skel = morphology.grey_dilation(skel, size=(3,3))


    # Scale the skel data to uint8 and add the outline mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skel = skel.astype(np.float32) # convert to float
        skel -= skel.min() # ensure the minimal value is 0.0
        if skel.max() != 0:
            skel /= skel.max() # maximum value in image is now 1.0
    tskel = np.uint8(cmap(skel)*255) # apply colormap to skel data.
    skel = gray2rgb(skel)
    skel[np.where(skel!=0)] = tskel[np.where(skel!=0)]

    if notMask:
        for i in range(3):
            skel[:,:,i] += outlines
    else:
        mask = gray2rgb(mask)
        skel = skel[:,:,:3]
        mask[np.where(skel!=0)] = skel[np.where(skel!=0)]
        skel = mask

    if returnSkel:
        return skel
    
    # Display the results.
    plt.imshow(skel, cmap = cmap, interpolation='none')
    plt.axis('off')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    if title is None:
        plt.gca().set_position([0, 0, 1, 1])
    else:
        plt.gca().set_position([0, 0, 1, 0.95])
        plt.title(title)
    plt.show()


def play_movie(frames, fps=10):
    '''
    Takes a list of frames and displays a movie
    Parameters:
    -----------
    frames : list of image arrays
        Image frames to be displayed as a movie
    fps : int
        Frames per second.
    '''
    print('Preparing movie...', end= ' ')
    start = time.time()
    fig = plt.figure()
    ims = []
    rows, cols, num_frames = frames.shape
    for i in range(num_frames):
        #update_progress(i/num_frames)
        ims.append([plt.imshow(frames[:,:,i], cmap=plt.cm.gray, animated=True,
                               vmin=0, vmax=255, interpolation='none')])
    ani = animation.ArtistAnimation(fig, ims, 1000/fps, True, 1000/fps)
    plt.axis('off')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.gca().set_position([0, 0, 1, 1])
    print('Done, took', round(time.time()-start,2), 'seconds.')
    plt.show()


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

def plot_grid(ims, titles=None, rows=0, cols=0, axis='off'):
    if titles is None:
        titles = ['' for _ in ims]
        
    if rows*cols < len(ims):
        sqrt = math.sqrt(len(ims))
        cols = int(sqrt) + 1
        rows = cols if cols * (cols - 1) < len(ims) else cols - 1
        
    fig, ax = plt.subplots(rows,cols)

    # Fill out rows or columns with empty images
    if rows*cols > len(ims):
        spaces = rows*cols - len(ims)
        for i in range(spaces):
            ims.append(np.ones((1,1)))
            titles.append('Empty')
        
    if rows > 1: # if more than one row
        for i in range(rows):
            for j in range(cols):
                n = i*cols + j
                if 'Empty' not in titles[n]:
                    ax[i,j].set_title(
                        ''.join((titles[n], ' (', str(n),')')))
                ax[i,j].axis(axis)
                ax[i,j].imshow(ims[n], cmap=plt.cm.gray, interpolation='none')
    else:
        for j in range(cols):
            ax[j].set_title(titles[j])
            ax[j].axis('off')
            ax[j].imshow(ims[j], cmap=plt.cm.gray, interpolation='none') 
    fig.tight_layout()
    plt.show()
    
def overlayMask(image, mask, color='o', return_overlay=False, animate=False,
                title=None, translucence=True):
    '''
    Displays the binary mask over the original image in order to verify results.
    
    Parameters
    ----------
    image : image array
        Image data prior to segmentation.
    mask : binary array
        Binary segmentation of the image data.  Must be the same size as image.
    color : str, optional
        The color of the overlaid mask.
    return_overlay : bool, optional
        If true, the image with the overlaid mask is returned and the overlay
        is not displayed here.
    animate : bool, optional
        If true, an animated figure will be displayed that alternates between
        showing the raw image and the image with the overlay.
    translucence : bool, option
        If True, the overlay will only change the color channels it needs too.
        

    Returns
    -------
    overlay : RGB image array, optional
        Color image with mask overlayyed on original image (only returned
        if 'return_overlay' is True).
    '''

    if title is None:
        title = 'Segmentation mask overlayed on image'

    img = np.copy(image)

    # Convert the image into 3 channels for a colored mask overlay
    overlay = gray2rgb(img)

    # Set color (default to blue if a proper color string is not given).
    r = 0
    g = 0
    b = 255
    if color == 'red' or color == 'r':
        r = 255
        g = 0
        b = 0
    if color == 'green' or color == 'g':
        r = 0
        g = 255
        b = 0
    if color == 'blue' or color == 'b':
        r = 0
        g = 0
        b = 255
    if color == 'white' or color == 'w':
        r = 255
        g = 255
        b = 255
    if color == 'yellow' or color == 'y':
        r = 255
        g = 255
        b = 0
    if color == 'orange' or color == 'o':
        r = 255
        g = 128
        b = 0
        
    # Apply mask.
    if not translucence or r != 0:
        overlay[mask == 1, 0] = r
    if not translucence or g != 0:
        overlay[mask == 1, 1] = g
    if not translucence or b != 0:
        overlay[mask == 1, 2] = b

    # Return or show overlay.
    if return_overlay:
        return overlay
    else:
        if animate:
            fig = plt.figure()
            ims = []
            for i in range(30):
                ims.append([plt.imshow(image, cmap=plt.cm.gray, animated=True)])
                ims.append([plt.imshow(overlay, animated=True)])
            ani = animation.ArtistAnimation(fig, ims, 1000, True, 1000)
            plt.axis('off')
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.gca().set_position([0, 0, 1, 0.95])
            plt.title(title)
            fig.canvas.set_window_title('Animated Mask Overlay')
            plt.show()            
        else:           
            showFull(overlay, title=title,interpolation='nearest')

        
