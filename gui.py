#!/usr/bin/env python3

# Author: Joshua Stuckner
# Date: 2017/06/21

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os
import glob
import numpy as np
import cv2
import time
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import string

from smart_tem import inout
from smart_tem import operations
from smart_tem import visualize


def user_input_good(input_string, allowed, boxName=''):

    # First make sure the box isn't empty
    if len(input_string) == 0:
        messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' value is missing.')))
        return False
    
    integer = ['integer', 'Int', 'int', 'Integer', '0']
    decimal = ['decimal', 'Dec', 'Decimal', 'dec', '2', 'float']
    signed_integer = ['sinteger', 'sInt', 'sint', 'sInteger', '1']

    if str(allowed) in integer:
        bad = ''.join((string.ascii_letters, string.punctuation,
                       string.whitespace))
        if any((c in bad) for c in input_string):
            messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' should be a posative integer value')))
            return False
        else:
            return True

    if str(allowed) in signed_integer:
        bad = ''.join((string.ascii_letters, string.punctuation,
                       string.whitespace))
        bad = ''.join(c for c in bad if c not in '-')
        if any((c in bad) for c in input_string) or '-' in input_string[1:]:
            messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' should be an integer value')))
            return False
        else:
            return True

    if str(allowed) in decimal:
        bad = ''.join((string.ascii_letters, string.punctuation,
                       string.whitespace))
        bad = ''.join(c for c in bad if c not in '.')
        if any((c in bad) for c in input_string) or input_string.count('.') > 1:
            messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' should be an posative decimal value')))
            return False
        else:
            return True
          

class selectRect(object):
    """
    Draws a rectangle on a figure and keeps track of the rectangle's size and
    location.  Used to select the target image data.

    Attributes
    ----------
    x0 : float64
        X coordinate (row) of start of rectangle.
    y0 : float 64
        Y coordinate (column) of start of rectangle.
    x1 : float64
        X coordinate (row) of end of rectangle.
    y1 : float 64
        Y coordinate (column) of end of rectangle.
    """

    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0, 0), 1, 1,
                              facecolor='none',
                              edgecolor='#6CFF33',
                              linewidth=3)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

def makeHDF5():
    makeApp = makeHDF5_GUI()
    makeApp.master.title('Convert to HDF5')
    makeApp.mainloop() 

class makeHDF5_GUI(tk.Toplevel):
    def __init__(self, master=None):
        tk.Toplevel.__init__(self, master)
        #self.pack()

        # Create tk variables
        self.varSelectRegion = tk.BooleanVar()
        self.varSubfolder = tk.BooleanVar()
        pad=5
        

        # Create widgits.
        self.butOk = tk.Button(self, text="OK", command=self.selectPaths,
                               width=8)
        self.butCancel = tk.Button(self, text="Cancel", command=self.cancel,
                                   width=8)
        self.checkSelect = tk.Checkbutton(self,
                text="Select region", variable=self.varSelectRegion)
        self.checkSubfolder = tk.Checkbutton(self,
                text="Include subfolders", variable=self.varSubfolder)

        # Pack widgets.
        self.butOk.grid(row=2, column=0, padx=pad, pady=pad)
        self.butCancel.grid(row=2, column=1, padx=pad, pady=pad)
        self.checkSelect.grid(row=0, column=0, columnspan=2, padx=pad, pady=3)
        self.checkSubfolder.grid(row=1, column=0, columnspan=2, padx=pad, pady=3)

    def cancel(self):
        self.destroy()

    def selectPaths(self):
        # Select the input folder.
        self.input_path = filedialog.askdirectory(
            title='Select the movie frame folder')
        if self.input_path == '':
            self.cancel()

        # Select output folder.
        self.output_path = filedialog.asksaveasfilename(defaultextension='.hdf5')
        if self.output_path == '':
            self.cancel()

        self.convert()

    def convert(self):
        ftypes = ['tif', 'jpg', 'tiff', 'bmp', 'png']
        # Get the image paths
        if self.varSubfolder.get():
            for ftype in ftypes:
                file_paths = glob.glob(
                    ''.join((self.input_path, '/**/*.', ftype)), recursive=True)
                if len(file_paths) > 0:
                    break
        else:
            for ftype in ftypes:
                file_paths = glob.glob(
                    ''.join((self.input_path, '/*.', ftype)))
                if len(file_paths) > 0:
                    break

        # Cancel if no files
        if len(file_paths) == 0:
            print('No valid files found.')
            self.cancel()
            time.sleep(0.1)

        # get image info
        im0 = inout.load(file_paths[0])
        rows, cols = im0.shape
        num_frames = len(file_paths)
        row_min = 0
        row_max = rows
        col_min = 0
        col_max = cols

        if self.varSelectRegion.get():
            # Get a sampling of the data for selection purposes
            sampling = np.zeros((rows,cols,10), dtype='uint8')
            mid = int(num_frames/2)
            for i in range(10):
                sampling[:,:,i] = inout.load(file_paths[mid+i])
                                       
            # Display a blurred sampling and select the target area
            sampling = operations.gaussian_stacking(sampling, 3)[:,:,5]
            sampling = operations.gaussian(sampling, 2)
            plt.axis('off')
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.gca().set_position([0, 0, 1, 1])
            plt.imshow(sampling, cmap=plt.cm.gray)
            a = selectRect()
            plt.show()

            # Clean selection data
            row_min = int(min(a.y0, a.y1))
            row_max = int(max(a.y0, a.y1)+1)
            col_min = int(min(a.x0, a.x1))
            col_max = int(max(a.x0, a.x1)+1)
            rows = row_max - row_min
            cols = col_max - col_min

        # Create numpy array
        frames = np.zeros((rows, cols, num_frames), dtype='uint8')
        print('Loading frames...')
        start = time.time()

        for i, path in enumerate(file_paths):
            if i % 10 == 0:
                visualize.update_progress(i/num_frames)
            image = inout.load(path)
            frames[:, :, i] = image[row_min:row_max, col_min:col_max]
        print('Done, took', round(time.time() - start, 2), 'seconds.')

        # Convert to hdf5 file
        print('Converting to hdf5 file...', end=' ')
        start = time.time()
        f = h5py.File(self.output_path, 'w')
        dset = f.create_dataset("data", (rows,cols,num_frames), compression='lzf', data=frames)
        f.close()
        print('Done, took', round(time.time() - start, 2), 'seconds.')

        self.cancel()
            
class MainApp(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()

        # Sizing variables
        pad=5
        spad=2
        butWidth = 15
        butHeight = 2

        # Data variables
        self.data = None
        self.rows = 0
        self.cols = 0
        self.num_frames = 0

        # Create frames.
        self.frameIO = tk.LabelFrame(self, text="Input/Output",
                                     width=200, height=100)
        self.frameVisualization = tk.LabelFrame(self, text="Visualization",
                                     width=200, height=100)
        self.frameOperations = tk.LabelFrame(self, text="Operations",
                                             width=200, height=100)

        # Create variables.
        # Operations
        self.var_slice_from = tk.StringVar()
        self.var_slice_to = tk.StringVar()
        self.varChambolleWeight = tk.StringVar()
        self.varChambolleStacking = tk.StringVar()
        
        # Create widgits.
        # IO frame
        self.butMakeHDF5 = tk.Button(self.frameIO, text="Images to HDF5",
                                     command=makeHDF5,
                                     width=butWidth, height=butHeight)
        self.butLoadHDF5 = tk.Button(self.frameIO, text="Load HDF5 file",
                                     command=self.loadHDF5,
                                     width=butWidth, height=butHeight)
        # Visualize frame
        self.butViewData = tk.Button(self.frameVisualization,
                                     text="Image viewer",
                                     command=self.viewData,
                                     width=butWidth, height=butHeight)
        self.butViewMovie = tk.Button(self.frameVisualization,
                                      text="Play Movie",
                                      command=self.playMovie,
                                      width=butWidth, height=butHeight)
        # Operations Frame.
        self.frame_slice = tk.Frame(self.frameOperations)
        self.but_slice = tk.Button(self.frame_slice,
                                      text="Slice dataset",
                                      command=self.slice,
                                      width=butWidth, height=1)
        self.lab_slice_from = tk.Label(self.frame_slice,
                                           text="From:")
        self.txt_slice_from = tk.Entry(self.frame_slice, width=5,
                                           textvariable=self.var_slice_from)
        self.lab_slice_to = tk.Label(self.frame_slice,
                                           text="To:")
        self.txt_slice_to = tk.Entry(self.frame_slice, width=5,
                                          textvariable=self.var_slice_to)
        self.frameChambolle = tk.Frame(self.frameOperations)
        self.butChambolle = tk.Button(self.frameChambolle,
                                      text="Chambolle denoise",
                                      command=self.chambolle,
                                      width=butWidth, height=1)
        self.labChambolleWeight = tk.Label(self.frameChambolle,
                                           text="Weight:")
        self.txtChambolleWeight = tk.Entry(self.frameChambolle, width=5,
                                           textvariable=self.varChambolleWeight)
        self.labChambolleStack = tk.Label(self.frameChambolle,
                                           text="Stacking:")
        self.txtChambolleStack = tk.Entry(self.frameChambolle, width=5,
                                          textvariable=self.varChambolleStacking)
                                           


        # Pack widgets.
        # Frames
        self.frameIO.grid(row=0, column=0, padx=pad, pady=pad)
        self.frameVisualization.grid(row=0, column=1, padx=pad, pady=pad)
        self.frameOperations.grid(row=1, column=0, padx=pad, pady=pad,
                                     columnspan=2)
        # IO frame
        self.butMakeHDF5.grid(row=0, column=0, padx=pad, pady=pad)
        self.butLoadHDF5.grid(row=1, column=0, padx=pad, pady=pad)
        # Visualize frame
        self.butViewData.grid(row=0, column=0, padx=pad, pady=pad)
        self.butViewMovie.grid(row=1, column=0, padx=pad, pady=pad)
        # Operations frame
        self.frame_slice.grid(row=0, column=0, padx=pad, pady=spad)
        self.lab_slice_from.grid(row=0, column=1, padx=pad, pady=pad)
        self.txt_slice_from.grid(row=0, column=2, padx=pad, pady=pad)
        self.lab_slice_to.grid(row=0, column=3, padx=pad, pady=pad)
        self.txt_slice_to.grid(row=0, column=4, padx=pad, pady=pad)
        self.but_slice.grid(row=0, column=0, padx=pad, pady=pad)
        
        self.frameChambolle.grid(row=1, column=0, padx=pad, pady=spad)
        self.labChambolleWeight.grid(row=0, column=1, padx=pad, pady=pad)
        self.txtChambolleWeight.grid(row=0, column=2, padx=pad, pady=pad)
        self.labChambolleStack.grid(row=0, column=3, padx=pad, pady=pad)
        self.txtChambolleStack.grid(row=0, column=4, padx=pad, pady=pad)
        self.butChambolle.grid(row=0, column=0, padx=pad, pady=pad)
        
        
    def loadHDF5(self):
        file = filedialog.askopenfilename(
            title='Select HDF5 file', defaultextension='.hdf5')
        self.data = inout.frames_from_hdf5(file)
        self.rows, self.cols, self.num_frames = self.data.shape
        self.saveShape()

    def saveShape(self):
        self.rows, self.cols, self.num_frames = self.data.shape
        
    def viewData(self):
        if self.data is None:
            print('No data is loaded.')
        else:
            v = visualize.Image_Viewer(self.data, main=False)

    def playMovie(self):
        if self.data is None:
            print('No data is loaded.')
        else:
            visualize.play_movie(self.data)


    def chambolle(self):
        if self.data is None:
            print('No data is loaded.')
        elif int(self.varChambolleStacking.get()) % 2 == 0 and \
             int(self.varChambolleStacking.get()) >= 0:
            messagebox.showwarning(
                "Input error",
                ''.join(('Stacking should be an odd integer')))
        elif user_input_good(self.varChambolleWeight.get(),
                           'decimal', 'Chambolle Weight') and \
            user_input_good(self.varChambolleStacking.get(),
                           'sint', 'Chambolle Stacking'):
            
            stacking = int(self.varChambolleStacking.get())
            stacking = None if stacking < 0 else stacking
            self.data = operations.tv_chambolle(
                self.data, weight=float(self.varChambolleWeight.get()),
                max_stacking=stacking)

    def slice(self):
        if self.data is None:
            print('No data is loaded.')
        elif user_input_good(self.var_slice_from.get(),
                             'int', 'From') and \
            user_input_good(self.var_slice_to.get(),
                            'int', 'To'):
            f = int(self.var_slice_from.get())
            t = int(self.var_slice_to.get())
            if t > self.num_frames:
                print('Input is out of range.')
            elif f >= t:
                print('To must be larger than From.')
            else:
                self.data = self.data[:,:,f:t]
                self.saveShape()
                print("Dataset sliced.")

            
    
##class OperationFrame(tk.Frame):
##
##    def __init__(self, parent, name, command, param1=None, param1type='int',
##                 param2=None, param2type='string'):
##
##        tk.Frame.__init__(self, parent)
##        self.name = name
##        self.param1 = param1
##        self.param2 = param2
##        self.param1type = param1type
##        self.param2type = param2type
    
if __name__ == "__main__":     
    myapp = MainApp()
    myapp.master.title('Denoise')
    myapp.mainloop() 
