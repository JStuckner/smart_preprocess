#!/usr/bin/env python3


# Author: Joshua Stuckner
# Date: 2017/06/21

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.font import Font
import os
import glob
import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import string

import inout
import operations
import visualize


DATA = None
UNDO = None

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def make_undo():
    global DATA
    global UNDO
    UNDO = DATA.copy()

def undo(data):
    global UNDO
    print('Undone.')
    temp = UNDO.copy()
    UNDO = data.copy()
    return temp
    
class CreateToolTip(object):
    '''
    create a tooltip for a given widget
    '''
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.close)
        self.timer = []
        self.tw = None
        
    def enter(self, event=None):
        self.timer = self.widget.after(700, self.display)
        
    def display(self):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background='#ffffe6', relief='solid', borderwidth=1,
                       font=("times", "10", "normal"))
        label.pack(ipadx=5, ipady=3)
        
    def close(self, event=None):
        if self.tw is not None:
            self.tw.destroy()
        self.widget.after_cancel(self.timer)

            
def user_input_good(input_string, allowed, boxName=''):
    """
    This function checks the input text and displays error messages if the input
    cannot be converted to the correct datatype.  Returns TRUE if the input is good.
    Returns FALSE if the input is not good.

    Params
    ======
    input_string: The string input by the end user.
    allowed (string): the datatype that is acceptable.

    Return
    ======
    bool: TRUE if the input is ok, FALSE if the input is not ok.
    """
    # First make sure the box isn't empty
    if len(input_string) == 0:
        messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' value is missing.')))
        return False
    
    integer = ['integer', 'Int', 'int', 'Integer', '0']
    decimal = ['decimal', 'Dec', 'Decimal', 'dec', '2', 'float']
    signed_integer = ['sinteger', 'sInt', 'sint', 'sInteger', '1']
    even_signed_integer = ['esint']

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

    if str(allowed) in even_signed_integer:
        bad = ''.join((string.ascii_letters, string.punctuation,
                       string.whitespace))
        bad = ''.join(c for c in bad if c not in '-')
        if any((c in bad) for c in input_string) or '-' in input_string[1:]:
            messagebox.showwarning(
                "Input error",
                ''.join((boxName, ' should be an integer value')))
            return False
        elif int(input_string) % 2 == 0 and int(input_string) >= 0:
            messagebox.showwarning(
                "Input error",
                ''.join(('Stacking should be an odd integer')))
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
            title='Select the image folder')
        if self.input_path == '':
            self.cancel()

        # Select output folder.
        self.output_path = filedialog.asksaveasfilename(defaultextension='.hdf5')
        if self.output_path == '':
            self.cancel()

        self.convert()

    def convert(self):
        ftypes = ['tif', 'jpg', 'tiff', 'bmp', 'png', 'dm4', 'dm3']
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
                try:
                    sampling[:,:,i] = inout.load(file_paths[mid+i])
                    last_good = i
                except IndexError:
                    sampling[:,:,i] = sampling[:,:,last_good]
                                       
            # Display a blurred sampling and select the target area
            sampling = operations.gaussian_stacking(sampling, 3)[:,:,5]
            sampling = operations.gaussian(sampling, 2)
            plt.axis('off')
            figManager = plt.get_current_fig_manager()
            try:
                figManager.window.showMaximized()
            except:
                figManager.window.state('zoomed')
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
        with h5py.File(self.output_path, 'w') as f:
            #dset = f.create_dataset("data", (rows,cols,num_frames), compression='lzf', data=frames)
            dset = f.create_dataset("data", (rows,cols,num_frames), data=frames)

        print('Done, took', round(time.time() - start, 2), 'seconds.')

        self.cancel()
            
class MainApp(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()

        global DATA

        # Sizing variables
        pad=5
        spad=2
        butWidth = 15
        butHeight = 2

        # Data variables
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
        self.butMakeHDF5 = tk.Button(self.frameIO, text="Create HDF5",
                                     command=makeHDF5,
                                     width=butWidth, height=butHeight)
        self.butLoadHDF5 = tk.Button(self.frameIO, text="Load HDF5",
                                     command=self.loadHDF5,
                                     width=butWidth, height=butHeight)
        self.butSaveMovie = tk.Button(self.frameIO, text="Save Movie",
                                     command=self.saveMovie,
                                     width=butWidth, height=butHeight)
        self.butSaveImages = tk.Button(self.frameIO, text="Save Images",
                                     command=self.saveImages,
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

        # Operation frames
        self.opsList = ['Data',
                        'Operations',
                        'Filters',
                        'Stacking',
                        'Other']
        self.opsDict = {}
        for i in self.opsList:
            self.opsDict[i] = tk.LabelFrame(self, text=i,width=500, height=100)
            
        # Operations Frame.
        self.ops = []
        
        self.ops.append(OperationFrame(
            self.opsDict['Data'], 'Undo',
            undo,
            help_text = None))
        
        self.ops.append(OperationFrame(
            self.opsDict['Data'], 'Slice dataset',
            slice_dataset,
            ['From:', 'To:'], ['int','int'], ['', ''],
            help_text = 'slice.txt'))
        
        self.ops.append(OperationFrame(
            self.opsDict['Operations'], 'Downsample',
            operations.down_sample,
            ['Pixel Size:', 'Nyquist:'], ['float','float'], ['', 3],
            help_text = 'downsample.txt'))
        
        self.ops.append(OperationFrame(
            self.opsDict['Operations'], 'Chambolle denoise',
            operations.tv_chambolle,
            ['Weight:', 'Stacks:'], ['float','sint'], [0.1, 1],
            help_text = 'chambolle.txt'))
        
        self.ops.append(OperationFrame(
            self.opsDict['Filters'], 'Gaussian Blur',
            operations.gaussian,
            ['Sigma:'], ['float'], [1],
            help_text = 'gaussian blur.txt'))

        self.ops.append(OperationFrame(
            self.opsDict['Filters'], 'Median Blur',
            operations.median,
            ['Size:'], ['int'], [3],
            help_text = 'median blur.txt'))

        self.ops.append(OperationFrame(
            self.opsDict['Filters'], 'Remove Outliers',
            operations.remove_outliers,
            ['Percent:', 'Size:'], ['float', 'int'], [0.1, 3],
            help_text = 'remove outliers.txt'))

        self.ops.append(OperationFrame(
            self.opsDict['Stacking'], 'Gaussian Stack',
            operations.gaussian_stacking,
            ['Sigma:'], ['float'], [1],
            help_text = 'gaussian stack.txt'))

        self.ops.append(OperationFrame(
            self.opsDict['Stacking'], 'Median Stack',
            operations.median_stacking,
            ['Frames:'], ['int'], [1],
            help_text = 'median stack.txt'))

        self.ops.append(OperationFrame(
            self.opsDict['Other'], 'Level Balance',
            operations.normalize,
            ['Clip:'], ['sint'], [3],
            help_text = 'contrast.txt'))

        self.ops.append(OperationFrame(
            self.opsDict['Other'], 'Unsharp Mask',
            operations.unsharp_mask,
            ['Sigma', 'Threshold:', 'Amount:'], ['float', 'float', 'float'],
            [1, 5, 0.5],
            help_text = 'unsharp.txt'))
        
                                           


        # Pack widgets.
        # Frames
        self.frameIO.grid(row=0, column=0, padx=pad, pady=pad)
        self.frameVisualization.grid(row=0, column=1, padx=pad, pady=pad)
##        self.frameOperations.grid(row=1, column=0, padx=pad, pady=pad,
##                                     columnspan=2, sticky='w')
        # IO frame
        self.butMakeHDF5.grid(row=0, column=0, padx=pad, pady=pad)
        self.butLoadHDF5.grid(row=1, column=0, padx=pad, pady=pad)
        self.butSaveMovie.grid(row=0, column=1, padx=pad, pady=pad)
        self.butSaveImages.grid(row=1, column=1, padx=pad, pady=pad)
        # Visualize frame
        self.butViewData.grid(row=0, column=0, padx=pad, pady=pad)
        self.butViewMovie.grid(row=1, column=0, padx=pad, pady=pad)
        #Operations frames
        for i, name in enumerate(self.opsList):
            self.opsDict[name].grid(row=1+i, column=0, padx=pad, pady=pad,
                                    columnspan=2, sticky='ew')
        # Operations subframes
        for i, op in enumerate(self.ops):
            op.grid(row=i, column=0, padx=pad, pady=spad, sticky='w')


        # Create tool tips
        ttp_makeHDF5 = CreateToolTip(self.butMakeHDF5,
            'Create an hdf5 file from a folder of images.')

        
    def loadHDF5(self):
        global DATA
        file = filedialog.askopenfilename(
            title='Select HDF5 file', defaultextension='.hdf5')
        DATA = inout.frames_from_hdf5(file)
        
    def viewData(self):
        global DATA
        if DATA is None:
            messagebox.showwarning(
                "Data Error",
                ''.join(('No data is loaded.')))
        else:
            v = visualize.Image_Viewer(DATA, main=False)

    def playMovie(self):
        global DATA
        if DATA is None:
            messagebox.showwarning(
                "Data Error",
                ''.join(('No data is loaded.')))
        else:
            visualize.play_movie(DATA)

    def saveMovie(self):
        global DATA
        if DATA is None:
            messagebox.showwarning(
                "Data Error",
                ''.join(('No data is loaded.')))
        else:
            path = filedialog.asksaveasfilename(defaultextension='.mp4')
            inout.save_movie(DATA, path)

    def saveImages(self):
        global DATA
        if DATA is None:
            messagebox.showwarning(
                "Data Error",
                ''.join(('No data is loaded.')))
        else:
            path = filedialog.askdirectory()
            inout.save_images(DATA, path)


def slice_dataset(data, f, t):
    _,_,num_frames = data.shape
    if t > num_frames:
        print('Input is out of range.')
    elif f >= t:
        print('To must be larger than From.')
    else:
        data = data[:,:,f:t]
        print("Dataset sliced.")
        return data

            

class OperationFrame(tk.Frame):
    global DATA
    def __init__(self, parent, name, command, params=[], paramtypes=[],
                 start_vals=None, sticky='w', help_text=None):
        """
        Creates a frame containing a button, some labels and entry boxes.
        The frame calls the appropriate command that is passed to it.
        The parameters of the command are the params in order.
        Paramtypes is the type of thing to put in the entrybox.
        Len(params) must equal len(paramtypes).
        """
        
        tk.Frame.__init__(self, parent)
        self.name = name
        self.params = params
        self.paramtypes = paramtypes
        self.command = command
        self.labels = []
        self.entrys = []
        self.vars = []
        self.help_text = help_text
        pad = 5
        spad = 2
        global DATA
                           

        # Create widgets
        self.button = tk.Button(self, text=self.name, width=15, height=1,
                                command=self.check_and_call)

        for param in params:
            self.vars.append(tk.StringVar())
            self.labels.append(tk.Label(self, text=param, width=8, anchor='e'))
            self.entrys.append(tk.Entry(self, textvariable=self.vars[-1],
                                        width=5))

        # Place widgets.
        self.button.grid(row=0, column=1, padx=pad, pady=spad, sticky='w')
        for i in range(len(self.labels)):
            self.labels[i].grid(row=0, column=2*i+2, padx=0, pady=spad, sticky='w')
            self.entrys[i].grid(row=0, column=2*i+3, padx=pad, pady=spad, sticky='w')


        # Fill in default values
        if start_vals is not None:
            for i, val in enumerate(start_vals):
                self.vars[i].set(str(val))

        # Create tool tip
        if self.help_text is not None:
            this_dir, this_filename = os.path.split(__file__)
            data_path = os.path.join(this_dir, "help", self.help_text)
            file = open(resource_path(data_path))
            data = file.read()
            file.close()
            ttp = CreateToolTip(self.button, data)

    def check_and_call(self):
        global DATA
        cleaned_params = []
        # error check
        good = []
        for var, paramtype in zip(self.vars, self.paramtypes):
            good.append(user_input_good(var.get(), paramtype, self.name))

        # types
        integer = ['integer', 'Int', 'int', 'Integer', '0']
        decimal = ['decimal', 'Dec', 'Decimal', 'dec', '2', 'float']
        signed_integer = ['sinteger', 'sInt', 'sint', 'sInteger', '1']
        esint = ['esint']

        if DATA is None:
            messagebox.showwarning(
                "Data Error",
                ''.join(('No data is loaded.')))
        # clean parameter
        elif all(good):
            for var, paramtype in zip(self.vars, self.paramtypes):
                if paramtype in integer or paramtype in signed_integer \
                        or paramtype in esint:
                    cleaned_params.append(int(var.get()))
                    #print(cleaned_params[-1])
                if paramtype in decimal:
                    cleaned_params.append(float(var.get()))

            # call the command
            if self.command != undo:
                make_undo()
            DATA = self.command(DATA, *cleaned_params)
            

            
if __name__ == "__main__":     
    myapp = MainApp()
    myapp.master.title('SMART-TEM')
    myapp.mainloop() 
