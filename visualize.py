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


from smart_preprocess import inout

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
    
        
