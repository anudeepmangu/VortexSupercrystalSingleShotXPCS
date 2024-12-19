#==============================================================================
# LCLS data visualization
# 2021-05-14
# 
# Ver. 6
# Yue Cao (ycao@colorado.edu)
#
#==============================================================================

# importing system packages
import os
import sys
import glob
import h5py
import time
import itertools

# importing the workhorse
import numpy as np
import pandas as pd
from scipy import io, signal, interpolate, ndimage

# tiff packages
import tifffile

# import all plotting packages
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
from mpl_toolkits.mplot3d import axes3d
from matplotlib import patches

# pyplot style
rcParams.update({'font.size': 14})
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.top'] = True
rcParams['ytick.right'] = True
rcParams['xtick.major.size'] = 10
rcParams['ytick.major.size'] = 10
rcParams['xtick.minor.size'] = 5
rcParams['ytick.minor.size'] = 5
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True


# import bokeh
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.palettes import Inferno256, Cividis256, inferno, cividis, viridis, Turbo256
from bokeh.models import LinearColorMapper, LogColorMapper, Div
from bokeh.layouts import gridplot, column, row

# import all fitting packages
from lmfit import Minimizer, Parameters, report_fit

#==============================================================================
# Global vars
#==============================================================================
# Path for saving the pngs from pyplot inline
# All plots from bokeh will be saved locally on the PC NOT the server
fig_dir = '/reg/data/ana16/xcs/xcslw6819/results/anudeep/Run 142/d'

#==============================================================================
# Slice ndarray along different dimensions
# All nan backfilled by 0.
#
# 0th dimension is the motor dimension for 3d datasets
#==============================================================================
def _val_to_idx(x, val):
    '''
    Internal methods for converting values to indices
    
    x:              1D array, monotonic
    val:            A value
    '''
    idx = np.nanargmin((x-val)**2)
    return idx


def getImg(imgStk, motor_pos, motor_val):
    '''
    Getting the image at a particular motor position
    
    Inputs:
    imgStk:     3D array
    motor_pos:  Motor name
    motor_val:  Motor value
    
    Outputs:
    img:        2D array
    '''
    motorID = _val_to_idx(motor_pos, motor_val)
    
    img = imgStk[motorID]
    return img


def normalize(imgStk, i0):
    '''
    imgStk:     3d array
    i0:         1d array
    '''
    if imgStk.shape[0]==i0.shape[0]:
        outStk = imgStk/i0[:,None,None]
        #outStk = np.nan_to_num(outStk, 0)
    else:
        print('***** Array size mistmatch. Abort i0 normalization. *****')
        outStk = imgStk
    
    return outStk


def getROIstk(imgStk, roi):
    '''
    imgStk:     3d array
    roi:        A list. [xmin, xmax, ymin, ymax]
    '''
    return imgStk[:, roi[2]:roi[3], roi[0]:roi[1]]


def getHcut(imgStk, yCen):
    '''
    imgStk:     3d array
    yCen:       An int. y position of horizontal cut
    '''
    return imgStk[:, yCen, :]


def getVcut(imgStk, xCen):
    '''
    imgStk:     3d array
    xCen:       An int. x position of vertical cut
    '''
    return imgStk[:, :, xCen]


#==============================================================================
# Summing ROIs
#==============================================================================
def sumROI(ROIstk):
    '''
    Return roi sum
    '''
    ROIsum = np.nansum(np.nansum(ROIstk, axis=2), axis=1)
    return ROIsum


def getROIcom(ROIstk):
    '''
    Return roi sum
    '''
    comx = []
    comy = []
    for i in range(ROIstk.shape[0]):
        (tempy, tempx) = ndimage.measurements.center_of_mass(ROIstk[i])
        comx.append(tempx)
        comy.append(tempy)
        
    comx = np.array(comx)
    comy = np.array(comy)
    return comx, comy


def sumCut(Cutstk, idMin=-1, idMax=-1):
    '''
    sumCut:     A Hstk or Vstk. Numpy array.
    idMin:      An int. Min of index for summation.
    idMax:      An int. Max of index for summation. 
    '''
    if idMin<0:
        idMin = 0
    if idMax<0:
        idMax = Cutstk.shape[1]
    Cutsum = np.nansum(Cutstk[:,idMin:idMax], axis=1)
    return Cutsum


#==============================================================================
# Conversions
#==============================================================================
def getChange(x, y, refXrange, norm=True):
    '''
    x:          1D array. The scanning motor
    y:          The signal
    refXrange:  A list. The reference range
    '''
    id0 = _val_to_idx(x, refXrange[0])
    id1 = _val_to_idx(x, refXrange[1])
    
    refy = np.nanmean(y[min(id0, id1):max(id0, id1)], axis=0)
    if len(y.shape)==1:
        if norm:
            yChange = (y-refy)/refy
        else:
            yChange = y-refy
    else:
        yChange = np.zeros(y.shape)
        if norm:
            for i in range(y.shape[0]):
                yChange[i] = (y[i]-refy)/refy
        else:
            for i in range(y.shape[0]):
                yChange[i] = y[i]-refy
    
    return yChange


def getFFT(x, y, refXrange, norm=True):
    '''
    x:          1D array. The scanning motor
    y:          The signal
    refXrange:  A list. The reference range
    '''
    yChange = getChange(x, y, refXrange, norm=norm)
    
    temp = np.fft.fft(yChange, axis=0)
    ffty = (np.real(temp)**2+np.imag(temp)**2)**0.5
    deltax = 1./np.abs(x[-1]-x[0])
    fftx = np.arange(len(x))*deltax
    
    return fftx, ffty


#==============================================================================
# Image operations
#==============================================================================
def rotate2D(img, xScale, yScale, CCWdeg, center, newhscale=[], newvscale=[], plotRot=False):
    '''
    Rotate CCW relative to a point in the 2D plane. Use negative value for CW rotation
    
    The coordinate transformation for the unit vectors:
                            |cos(deg)   -sin(deg)|
    (e_x' e_y') = (e_x e_y) |                    |
                            |sin(deg)    cos(deg)|
    And for the coordinates:
    
    |x'|   |cos(deg)   -sin(deg)| |x|
    |  | = |                    | | |
    |y'|   |sin(deg)    cos(deg)| |y|
    
    With a simple for loop, 3D rotation could be performed.
    
    Inputs:
    CCWdeg:         In degrees and has to be in [-90 deg, 90 deg]
    center:         The center of rotation. A list.
    newhscale:      Optional. Default is []. For a customized new horizontal scale
                    for output. The format is [vmin, vmax, num]
    plotRot:        Optional. Whether or not to compare the 'before' and 'after'
    '''
    # ndimage.rotate(img2D, deg_in_pix) will not work. 
    # The shape of the rotated 2D img and the rotated scales
    # cannot be determined easily
    
    xCenter = center[0]
    yCenter = center[1]

    # If we only want to plot a figure rotated, we then just need rotxx, rotyy
    xx, yy = np.meshgrid(xScale, yScale)
    rotxx = np.cos(CCWdeg*np.pi/180)*(xx-xCenter)-np.sin(CCWdeg*np.pi/180)*(yy-yCenter)+xCenter
    rotyy = np.sin(CCWdeg*np.pi/180)*(xx-xCenter)+np.cos(CCWdeg*np.pi/180)*(yy-yCenter)+yCenter

    # As the rotxx, rotyy are not linear grids, we need to flatten them into (x,y) pairs

    points = np.vstack((rotxx.flatten(), rotyy.flatten())).transpose()
    values = img.flatten()

    # Setting up the new grid
    rotXmin = rotxx.min()
    rotXmax = rotxx.max()
    rotYmin = rotyy.min()
    rotYmax = rotyy.max()
    numX = int(np.abs((rotXmax-rotXmin)/(xScale[1]-xScale[0]))+1)
    numY = int(np.abs((rotYmax-rotYmin)/(yScale[1]-yScale[0]))+1)
    newX = np.linspace(rotXmin, rotXmax, num=numX)
    newY = np.linspace(rotYmin, rotYmax, num=numY)

    if not newhscale==[]:
        newX = np.linspace(newhscale[0], newhscale[1], num=newhscale[2])

    if not newvscale==[]:
        newY = np.linspace(newvscale[0], newvscale[1], num=newvscale[2])
    
    newXX, newYY = np.meshgrid(newX, newY)

    # Interpolate
    newImg = interpolate.griddata(points, values, (newXX, newYY), method='cubic')
    
    if plotRot:
        fig, ax = plt.subplots(1,3)
        ax[0].pcolormesh(xx, yy, img)
        ax[0].set_aspect(1)
        ax[0].set_title('Before')

        ax[1].pcolormesh(rotxx, rotyy, img)
        ax[1].set_aspect(1)
        ax[1].set_title('After')

        ax[2].pcolormesh(newXX, newYY, newImg)
        ax[2].set_aspect(1)
        ax[2].set_title('Interped')

        fig.suptitle('CCW rotation {} deg'.format(CCWdeg))
        fig.tight_layout()
    
    return newImg, newX, newY


#==============================================================================
# Displays - lines
#==============================================================================
def displayLine(x, y, titleStr='', xAxStr='', yAxStr='', xyScale='linear', mode='interactive', outfig=False):
    '''
    Display a single line
    The default mode is interactive
    
    Inputs:
    x:              1D array
    y:              1D array
    
    titleStr:       Title of the fig
    xAxStr:         X axis label
    yAxStr:         Y axis label
    
    mode:           If 'interactive', bokeh will be used.
                    Otherwise, pyplot inline will be used.
                    
    outfig:         Saving the fig as png. Only works for the 
                    pyplot inline.
    '''
    if mode=='interactive':
        # Using bokeh
        output_notebook()
        if xyScale=='semilogx':
            fig = figure(width=400, height=300, x_axis_type='log', toolbar_location='right', tooltips=[("(x,y)", "($x, $y)")])
        elif xyScale=='semilogy':
            fig = figure(width=400, height=300, y_axis_type='log', toolbar_location='right', tooltips=[("(x,y)", "($x, $y)")])
        elif xyScale=='loglog':
            fig = figure(width=400, height=300, x_axis_type='log', y_axis_type='log', toolbar_location='right', tooltips=[("(x,y)", "($x, $y)")])
        else:
            fig = figure(width=400, height=300, toolbar_location='right', tooltips=[("(x,y)", "($x, $y)")])
        fig.circle(x, y)
        fig.line(x, y)

        if not xAxStr=='':
            fig.xaxis.axis_label = xAxStr

        if not yAxStr=='':
            fig.yaxis.axis_label = yAxStr

        if not titleStr=='':
            fig.title.text = titleStr
            fig.title.align = 'center'

        show(fig)
    else:
        # Using plt
        plt.figure()
        if xyScale=='semilogx':
            plt.semilogx(x, y, 'o-')
        elif xyScale=='semilogy':
            plt.semilogy(x, y, 'o-')
        elif xyScale=='loglog':
            plt.loglog(x, y, 'o-')
        else:
            plt.plot(x, y, 'o-')
        
        if not xAxStr=='':
            plt.xlabel(xAxStr)
        if not yAxStr=='':
            plt.ylabel(yAxStr)
        if not titleStr=='':
            plt.title(titleStr)
        plt.tight_layout()
        
        if outfig:
            if titleStr=='':
                titleStr='out'
            plt.savefig(os.path.join(fig_dir, titleStr+'.png'))
        
    return


def displayLines(xs, ys, labels=[], titleStr='', xAxStr='', yAxStr='', xyScale='linear', mode='interactive', outfig=False):
    '''
    Display multiple lines on the same plot.
    The default mode is interactive
    
    Inputs:
    x:              A list of 1D array
    y:              A list of 1D array
    
    titleStr:       Title of the fig
    xAxStr:         X axis label
    yAxStr:         Y axis label
    
    mode:           If 'interactive', bokeh will be used.
                    Otherwise, pyplot inline will be used.
                    
    outfig:         Saving the fig as png. Only works for the 
                    pyplot inline.
    '''
    # Checking list length
    if not len(xs)==len(ys):
        print('Warning: x and y are not paired.')
    else:
        if mode=='interactive':
            # Using bokeh
            color_mapper=cividis(len(xs))

            output_notebook()
            if xyScale=='semilogx':
                fig = figure(width=400, height=300, x_axis_type='log', toolbar_location='right', tooltips=[("(x,y)", "($x, $y)")])
            elif xyScale=='semilogy':
                fig = figure(width=400, height=300, y_axis_type='log', toolbar_location='right', tooltips=[("(x,y)", "($x, $y)")])
            elif xyScale=='loglog':
                fig = figure(width=400, height=300, x_axis_type='log', y_axis_type='log', toolbar_location='right', tooltips=[("(x,y)", "($x, $y)")])
            else:
                fig = figure(width=400, height=300, toolbar_location='right', tooltips=[("(x,y)", "($x, $y)")])

            if len(labels):
                if len(labels)==len(xs):
                    for i in range(len(xs)):
                        fig.circle(xs[i], ys[i], fill_color=color_mapper[i], line_color=color_mapper[i], 
                                   legend_label=labels[i])
                        fig.line(xs[i], ys[i], line_color=color_mapper[i], legend_label=labels[i])
                else:
                    print('Warning: label number incorrect.')
                    for i in range(len(xs)):
                        fig.circle(xs[i], ys[i], fill_color=color_mapper[i], line_color=color_mapper[i])
                        fig.line(xs[i], ys[i], line_color=color_mapper[i])
            else:
                for i in range(len(xs)):
                    fig.circle(xs[i], ys[i], fill_color=color_mapper[i], line_color=color_mapper[i])
                    fig.line(xs[i], ys[i], line_color=color_mapper[i])

            if not xAxStr=='':
                fig.xaxis.axis_label = xAxStr

            if not yAxStr=='':
                fig.yaxis.axis_label = yAxStr

            if not titleStr=='':
                fig.title.text = titleStr
                fig.title.align = 'center'

            show(fig)
        else:
            # Using plt
            # The total number of lines plotted
            numLines = len(xs)

            # Generating a color table
            color=iter(cm.rainbow(np.linspace(0,1,numLines)))
            
            plt.figure()
            plt.clf()
            if len(labels):
                if len(labels)==len(xs):
                    for i in range(len(xs)):
                        if xyScale=='semilogx':
                            plt.semilogx(xs[i], ys[i], 'o-', color=next(color), label=str(labels[i]))
                        elif xyScale=='semilogy':
                            plt.semilogy(xs[i], ys[i], 'o-', color=next(color), label=str(labels[i]))
                        elif xyScale=='loglog':
                            plt.loglog(xs[i], ys[i], 'o-', color=next(color), label=str(labels[i]))
                        else:
                            plt.plot(xs[i], ys[i], 'o-', color=next(color), label=str(labels[i]))
                    plt.legend()
                else:
                    print('Warning: label number incorrect.')
                    for i in range(len(xs)):
                        if xyScale=='semilogx':
                            plt.semilogx(xs[i], ys[i], 'o-', color=next(color))
                        elif xyScale=='semilogy':
                            plt.semilogy(xs[i], ys[i], 'o-', color=next(color))
                        elif xyScale=='loglog':
                            plt.loglog(xs[i], ys[i], 'o-', color=next(color))
                        else:
                            plt.plot(xs[i], ys[i], 'o-', color=next(color))
            else:
                for i in range(len(xs)):
                    
                    if xyScale=='semilogx':
                        plt.semilogx(xs[i], ys[i], 'o-', color=next(color))
                    elif xyScale=='semilogy':
                        plt.semilogy(xs[i], ys[i], 'o-', color=next(color))
                    elif xyScale=='loglog':
                        plt.loglog(xs[i], ys[i], 'o-', color=next(color))
                    else:
                        plt.plot(xs[i], ys[i], 'o-', color=next(color))

            if not xAxStr=='':
                plt.xlabel(xAxStr)
            if not yAxStr=='':
                plt.ylabel(yAxStr)
            if not titleStr=='':
                plt.title(titleStr)
            plt.tight_layout()
            
            if outfig:
                if titleStr=='':
                    titleStr='out'
                plt.savefig(os.path.join(fig_dir, titleStr+'.png'))
        
    return


def stackLines(x, ys, titleStr='', xAxStr='', yAxStrs=[], mode='interactive', outfig=False):
    '''
    Stacking lines by sharing the same x axis, and one line per panel
    '''
    if mode=='interactive':
        # Using bokeh
        color_mapper=cividis(len(ys))
        output_notebook()

        f1 = figure(width=400, height=200, tooltips=[("(x,y)", "($x, $y)")])
        f1.xaxis.visible = False
        f1.circle(x, ys[0], fill_color=color_mapper[0], line_color=color_mapper[0])
        f1.line(x, ys[0], line_color=color_mapper[0])

        if len(yAxStrs):
            f1.yaxis.axis_label = yAxStrs[0]

        if not titleStr=='':
            f1.title.text = titleStr
            f1.title.align = 'center'

        gridlist = [[f1]]

        for i in range(1, len(ys)):
            f = figure(width=400, height=200, x_range=f1.x_range, tooltips=[("(x,y)", "($x, $y)")])
            f.circle(x, ys[i], fill_color=color_mapper[i], line_color=color_mapper[i])
            f.line(x, ys[i], line_color=color_mapper[i])
            if len(yAxStrs):
                f.yaxis.axis_label = yAxStrs[i]

            gridlist.append([f])

        if not xAxStr=='':
            # Only label the x axis of the last panel
            f.xaxis.axis_label = xAxStr

        fig = gridplot(gridlist)
        show(fig)
    else:
        # Using plt
        # The total number of lines plotted.
        numLines = len(ys)

        # Generating a color table
        color=iter(cm.rainbow(np.linspace(0,1,numLines)))
            
        fig, axes = plt.subplots(len(ys), 1, sharex=True, figsize=(8, 2.5*len(ys)))
        for i in range(len(ys)):
            axes[i].plot(x, ys[i], 'o-', color=next(color))
        
        if not xAxStr=='':
            axes[-1].set_xlabel(xAxStr)
        if len(yAxStrs):
            if len(yAxStrs)==len(ys):
                for i in range(len(ys)):
                    axes[i].set_ylabel(yAxStrs[i])
            else:
                print('Warning: label number incorrect.')
        if not titleStr=='':
            fig.suptitle(titleStr)
        fig.tight_layout()
            
        if outfig:
            if titleStr=='':
                titleStr='out'
            plt.savefig(os.path.join(fig_dir, titleStr+'.png'))
        
    return


# def displayFFT(x, y, refXrange, titleStr='', xAxStr='', yAxStr=''):
#     '''
#     Display the FFT of (x, y)
    
#     x:          An array. Has to be linear
#     refXrange:  A list. The reference range
#     '''
#     fftx, ffty = getFFT(x, y, refXrange)
    
#     output_notebook()
#     f1 = figure(width=300, height=200, tooltips=[("(x,y)", "($x, $y)")])
#     f1.circle(x, y, fill_color='orange', line_color='orange')
#     f1.line(x, y, line_color='orange')

#     f2 = figure(width=300, height=200, tooltips=[("(x,y)", "($x, $y)")])
#     f2.circle(fftx[1:], ffty[1:], fill_color='purple', line_color='purple')
#     f2.line(fftx[1:], ffty[1:], line_color='purple')

#     if not xAxStr=='':
#         f1.xaxis.axis_label = xAxStr

#     if not yAxStr=='':
#         f1.yaxis.axis_label = yAxStr

#     f2.yaxis.axis_label = 'FFT '+yAxStr

#     if not titleStr=='':
#         f1.title.text = titleStr
#         f1.title.align = 'center'

#     fig = gridplot([[f1, f2]])
#     show(fig)
    

#==============================================================================
# Displays - images
#==============================================================================
def displayImg(img, colorScale='linear', vmin=None, vmax=None, titleStr='', mode='interactive', outfig=''):
    if vmin==None:
        vmin = max(np.nanpercentile(img, 2), 1e-16)
        
    if vmax==None:
        vmax = np.nanpercentile(img, 98)
        
    if mode=='interactive':
        # Using bokeh
        if colorScale=='linear':
            color_mapper = LinearColorMapper(palette=Turbo256, low=vmin, high=vmax)
        if colorScale=='log':
            color_mapper = LogColorMapper(palette=Turbo256, low=vmin, high=vmax)

        output_notebook()
        fig = figure(width=400, height=int(400*img.shape[0]/img.shape[1]), tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        fig.x_range.range_padding = 0
        fig.y_range.range_padding = 0

        # must give a vector of image data for image parameter
        fig.image(image=[img], x=-0.5, y=-0.5, dw=img.shape[1], dh=img.shape[0], color_mapper=color_mapper, level="image")
        fig.grid.grid_line_width = 0.5

        if not titleStr=='':
            fig.title.text = titleStr
            fig.title.align = 'center'

        show(fig)
    else:
        # Using plt
        plt.figure()
        if colorScale=='linear':
            plt.imshow(img, vmin=vmin, vmax=vmax,origin='lower')
            plt.colorbar()
        if colorScale=='log':
            plt.imshow(img, norm=colors.LogNorm(vmin=vmin, vmax=vmax),origin='lower')
            plt.colorbar()
        
        if not titleStr=='':
            plt.title(titleStr)
        plt.tight_layout()
            
        if not outfig=='':
            if titleStr=='':
                titleStr='out'
            plt.savefig(os.path.join(outfig, titleStr+'.png'))
        
    return


def showROI(img, roiList, colorScale='linear', vmin=None, vmax=None, titleStr='', mode='interactive', outfig=''):
    '''
    [[xmin1, xmax1, ymin1, ymax1], [xmin2, xmax2, ymin2, ymax2]]
    '''
    if vmin==None:
        vmin = max(np.nanpercentile(img, 2), 1e-16)
        
    if vmax==None:
        vmax = np.nanpercentile(img, 98)
        
    if mode=='interactive':
        # Using bokeh
        if colorScale=='linear':
            color_mapper = LinearColorMapper(palette=Turbo256, low=vmin, high=vmax)
        if colorScale=='log':
            color_mapper = LogColorMapper(palette=Turbo256, low=vmin, high=vmax)

        output_notebook()
        fig = figure(width=400, height=int(400*img.shape[0]/img.shape[1]), tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        fig.x_range.range_padding = 0
        fig.y_range.range_padding = 0

        # must give a vector of image data for image parameter
        fig.image(image=[img], x=-0.5, y=-0.5, dw=img.shape[1], dh=img.shape[0], color_mapper=color_mapper, level="image")
        fig.grid.grid_line_width = 0.5

        # Draw ROIs
        # If we want to define a new color wheel
        colorList = iter(['deepskyblue', 'cyan', 'lime', 'greenyellow', 'orange', 'pink'])

        # Using inverted viridis
        #colorList = viridis(len(roiList)*2)
        #colorList = colorList[::-1]

        for i, roi in enumerate(roiList):
            tempx = [roi[0], roi[1], roi[1], roi[0], roi[0]]
            tempy = [roi[2], roi[2], roi[3], roi[3], roi[2]]
            # Using the newly defined color wheel
            fig.line(tempx, tempy, line_color=next(colorList), legend_label='ROI {}'.format(i+1))

            # Using inverted viridis
            #fig.line(tempx, tempy, line_color=colorList[i], legend_label='ROI {}'.format(i+1))

        if not titleStr=='':
            fig.title.text = titleStr
            fig.title.align = 'center'

        show(fig)
    else:
        # Using plt
        plt.figure()
        if colorScale=='linear':
            plt.imshow(img, vmin=vmin, vmax=vmax,origin='lower')
            plt.colorbar()
        if colorScale=='log':
            plt.imshow(img, norm=colors.LogNorm(vmin=vmin, vmax=vmax),origin='lower')
            plt.colorbar()
            
        numLines = len(roiList)
        color=iter(cm.Reds(np.linspace(0,1,numLines)))
        for i, roi in enumerate(roiList):
            tempx = [roi[0], roi[1], roi[1], roi[0], roi[0]]
            tempy = [roi[2], roi[2], roi[3], roi[3], roi[2]]
            plt.plot(tempx, tempy, '-', color=next(color), label='ROI {}'.format(i+1))
            plt.legend()
        if not titleStr=='':
            plt.title(titleStr)
        plt.tight_layout()
            
        if not outfig=='':
            if titleStr=='':
                titleStr='out'
            plt.savefig(os.path.join(outfig, titleStr+'.png'))
        
    return


def compareImg(img1, img2, colorScale='linear', vmin=None, vmax=None, vdiffmin=None, vdiffmax=None,
               titleStrs=[], mode='interactive', outfig=False):
    '''
    The colorScale only applies to the first two images, not their differences.
    '''
    if vmin==None:
        vmin = max(np.nanpercentile(img1, 2), 1e-16)
    if vmax==None:
        vmax = np.nanpercentile(img1, 98)
        
    diffimg = img2-img1
    if vdiffmin==None:
        vdiffmin = np.nanpercentile(diffimg, 2)
    if vdiffmax==None:
        vdiffmax = np.nanpercentile(diffimg, 98)
    
    if mode=='interactive':
        # Using bokeh
        if colorScale=='linear':
            color_mapper = LinearColorMapper(palette=Inferno256, low=vmin, high=vmax)
        if colorScale=='log':
            color_mapper = LogColorMapper(palette=Inferno256, low=vmin, high=vmax)
        diffcolor_mapper = LinearColorMapper(palette=Inferno256, low=vdiffmin, high=vdiffmax)

        output_notebook()
        f1 = figure(width=300, height=int(300*img1.shape[0]/img1.shape[1]), tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        f1.x_range.range_padding = 0
        f1.y_range.range_padding = 0
        f1.image(image=[img1], x=-0.5, y=-0.5, dw=img1.shape[1], dh=img1.shape[0], 
                 color_mapper=color_mapper, level="image")
        f1.grid.grid_line_width = 0.5

        f2 = figure(width=300, height=int(300*img2.shape[0]/img2.shape[1]), x_range=f1.x_range, y_range=f1.y_range, 
                    tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        f2.x_range.range_padding = 0
        f2.y_range.range_padding = 0
        f2.image(image=[img2], x=-0.5, y=-0.5, dw=img2.shape[1], dh=img2.shape[0], 
                 color_mapper=color_mapper, level="image")
        f2.grid.grid_line_width = 0.5

        f3 = figure(width=300, height=int(300*img1.shape[0]/img1.shape[1]), x_range=f1.x_range, y_range=f1.y_range, 
                    tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        f3.x_range.range_padding = 0
        f3.y_range.range_padding = 0
        f3.image(image=[diffimg], x=-0.5, y=-0.5, dw=img2.shape[1], dh=img2.shape[0], 
                 color_mapper=diffcolor_mapper, level="image")
        f3.grid.grid_line_width = 0.5

        if len(titleStrs)==2:
            f1.title.text = titleStrs[0]
            f1.title.align = 'center'

            f2.title.text = titleStrs[1]
            f2.title.align = 'center'

            f3.title.text = '{} - {}'.format(titleStrs[1], titleStrs[0])
            f3.title.align = 'center'

        fig = gridplot([[f1, f2, f3]])

        show(fig)
    else:
        # Using plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        if colorScale=='linear':
            im1 = axes[0].imshow(img1, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im1, cax=cax)
            
            im2 = axes[1].imshow(img2, vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im2, cax=cax)
            
            im3 = axes[2].imshow(diffimg, vmin=vdiffmin, vmax=vdiffmax)
            divider = make_axes_locatable(axes[2])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im3, cax=cax)
        if colorScale=='log':
            im1 = axes[0].imshow(img1, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im1, cax=cax)
            
            im2 = axes[1].imshow(img2, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im2, cax=cax)
            
            im3 = axes[2].imshow(diffimg, vmin=vdiffmin, vmax=vdiffmax)
            divider = make_axes_locatable(axes[2])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im3, cax=cax)
            
        if len(titleStrs)==2:
            axes[0].set_title(titleStrs[0])
            axes[1].set_title(titleStrs[1])
            axes[2].set_title('{} - {}'.format(titleStrs[1], titleStrs[0]))
            
        fig.tight_layout()
        
        if outfig:
            if titleStrs==[]:
                plt.savefig(os.path.join(fig_dir, 'cf.png'))
            else:
                plt.savefig(os.path.join(fig_dir, '{}-{}-cf'.format(titleStrs[0], titleStrs[1])+'.png'))
            
    return