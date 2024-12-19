#==============================================================================
# 2D fitting
# 2020-05-20
# 
# Ver. 1
# Yue Cao (ycao@colorado.edu)
#
# General comments:
#
# 1) In all the inputs, the x and y are 1D, and the z is 2D.
#
# 2) We could put the actual scales of x and y (e.g. degrees, A^-1) as inputs.
# The outputs will be scaled accordingly.
#
# 3) The ROI is in pixels and not scaled, in the format of [xmin, xmax, ymin, ymax]
#
# For each fitting, there are 2 parts:
# The function for the model, the function making the initial guess
# Note the misfit from the minimizer is a flattened 2D array, 
# and will have to be reshaped for comparison.
#
#
#==============================================================================

# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 100

import matplotlib.pyplot as plt
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
from scipy import io, signal, interpolate

# tiff packages
import tifffile

from lmfit import Minimizer, Parameters, report_fit

#==============================================================================
# Internal methods for converting values to indices
#==============================================================================

def _val_to_idx(x, val):
    '''
    Internal methods for converting values to indices
    
    x:              1D array, monotonic
    val:            A value
    '''
    idx = np.nanargmin((x-val)**2)
    
    return idx


def _lim_to_bounds(x, ROI):
    '''
    Internal methods for converting values to indices
    
    x:              1D array, monotonic
    ROI:            A list in the format [xmin, xmax]
    '''
    idx0 = np.nanargmin((x-ROI[0])**2)
    idx1 = np.nanargmin((x-ROI[1])**2)
    idmin = np.min([idx0, idx1])
    if idmin<0:
        idmin = 0
    idmax = np.max([idx0, idx1])
    if idmax>len(x)-1:
        idmax = len(x)-1
    
    return idmin, idmax


#==============================================================================
# General fitting methods
#==============================================================================

def do_fit2D(x, y, z, model_func, guess_func, params=[], ROI=[], verbose=False, plotFit=False):
    '''
    General 2D fit func
    
    One could choose to put the initial guess by hand, or to use
    the automatic guess
    Inputs:
    x: 1D
    y: 1D
    z: 2D
    '''
    if not ROI==[]:
        xidmin, xidmax = _lim_to_bounds(x, [ROI[0], ROI[1]])
        yidmin, yidmax = _lim_to_bounds(y, [ROI[2], ROI[3]])
        tx = x[xidmin:xidmax]
        ty = y[yidmin:yidmax]
        tz = z[yidmin:yidmax, xidmin:xidmax]
    else:
        tx = x
        #print(tx)
        ty = y
        #print(ty)
        tz = z
        #print(tz)
    
    if params==[]:
        params = guess_func(tx, ty, tz)
        
    minner = Minimizer(model_func, params, fcn_args=(tx, ty, tz))
    result = minner.minimize()
    
    misfit = result.residual.reshape(tz.shape)*(-1)
    # fitted = tz + result.residual.reshape(tz.shape)
    fitted = tz-misfit
    chisqr = result.redchi
    
    # Getting all parameters
    fit_params = pd.DataFrame(index=params.keys(), columns=['value', 'err']).astype(float)
    for key in params.keys():
        fit_params['value'].loc[key] = result.params[key].value
        fit_params['err'].loc[key] = result.params[key].stderr

    if verbose:
        report_fit(result)
    
    if plotFit:
        fig, ax = plt.subplots(1,3)
        im0 = ax[0].imshow(tz,cmap='inferno')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im0, cax=cax)
        ax[0].set_title('Raw data')
        
        im1 = ax[1].imshow(fitted,cmap='inferno')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax)
        ax[1].set_title('Fit')
        im2 = ax[2].imshow(misfit,cmap='inferno')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax)
        ax[2].set_title('Misfit')
        fig.tight_layout()
        
        fig, ax = plt.subplots(1,2)
        ax[0].plot(tz.sum(axis=1), 'b.', label='Raw')
        ax[0].plot(fitted.sum(axis=1), 'r-', label='Fit')
        ax[0].plot(misfit.sum(axis=1), 'k--', label='Misfit')
        ax[0].legend()
        ax[0].set_xlabel('x')
        ax[0].set_title('Summed along y')
        ax[1].plot(tz.sum(axis=0), 'b.', label='Raw')
        ax[1].plot(fitted.sum(axis=0), 'r-', label='Fit')
        ax[1].plot(misfit.sum(axis=0), 'k--', label='Misfit')
        ax[1].legend()
        ax[1].set_xlabel('y')
        ax[1].set_title('Summed along x')
        fig.tight_layout()
    
    return fit_params, tx, ty, tz, fitted, misfit, chisqr


def fitted_to_params2D(fit_params):
    '''
    Transferring fitted
    values into the guess of another fit
    
    The bounds and constraints will not be transferred
    '''
    params = Parameters()
    for key in fit_params.index:
        params.add(key, value=fit_params['value'].loc[key])
        
    return params


#==============================================================================
# Fitting funcs
# Basic fitting funcs
# Ref: https://gpufit.readthedocs.io/en/latest/fit_model_functions.html
#
# Anisotropic Gaussian
# https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
#==============================================================================
def Gauss2D(params, x, y, z):
    '''
    2D Gaussian
    
    Input:
    x: x coordinate, 1D
    y: y coordinate, 1D
    z: height values
    '''
    A = params['Area']
    sx = params['sigma_x']
    sy = params['sigma_y']
    xc = params['cen_x']
    yc = params['cen_y']
    
    bg_x = params['BG_slope_x']
    bg_y = params['BG_slope_y']
    bg_c = params['BG_const']
    
    xx, yy = np.meshgrid(x, y)
    
    model = A*np.exp(-(xx-xc)**2/2/sx**2-(yy-yc)**2/2/sy**2)/(2*np.pi*sx*sy)
    model = model+bg_x*xx+bg_y*yy+bg_c
    
    return model-z


def guess_Gauss2D(x, y, z):
    '''
    Making initial guess
    x, y can be identified as pixel number, or if in need, HKL values
    '''
    xx, yy = np.meshgrid(x, y)
    
    # Guessing the center
    cen = np.unravel_index(z.argmax(), z.shape)
    yc = y[cen[0]]
    xc = x[cen[1]]
    
    # Guessing the BG
    bg_c = np.nanmin(z)
    height = np.nanmax(z)-bg_c
    
    # Guessing the widths and BG slopes
    temp = z[cen[0]]
    bg_x = (temp[-1]-temp[0])/(x[-1]-x[0])
    templeft = temp[:cen[1]]
    sx_left = np.argwhere(templeft<bg_c+height/2)[-1,0]
    tempright = temp[cen[1]:]
    sx_right = np.argwhere(tempright<bg_c+height/2)[0,0]
    sx = np.abs((x[sx_right]-x[sx_left])/2.)
    
    temp = z[:, cen[1]]
    bg_y = (temp[-1]-temp[0])/(y[-1]-y[0])
    templeft = temp[:cen[0]]
    sy_left = np.argwhere(templeft<bg_c+height/2)[-1,0]
    tempright = temp[cen[0]:]
    sy_right = np.argwhere(tempright<bg_c+height/2)[0,0]
    sy = np.abs((y[sy_right]-y[sy_left])/2.)
    
    # Guessing the area
    A = 2*np.pi*sx*sy*height
    
    params = Parameters()
    params.add('Area', value=A)
    params.add('sigma_x', value=sx)
    params.add('sigma_y', value=sy)
    params.add('cen_x', value=xc)
    params.add('cen_y', value=yc)
    params.add('BG_slope_x', value=bg_x)
    params.add('BG_slope_y', value=bg_y)
    params.add('BG_const', value=bg_c)
    
    return params


#==============================================================================
# Rotated anisotropic Gaussian
#==============================================================================
def RotGauss2D(params, x, y, z):
    '''
    Rotated 2D Gaussian
    
    Input:
    x: x coordinate, 1D
    y: y coordinate, 1D
    z: height values
    '''
    A = params['Area']
    sx = params['sigma_x']
    sy = params['sigma_y']
    xc = params['cen_x']
    yc = params['cen_y']
    th = params['theta']*np.pi/180.
    
    bg_x = params['BG_slope_x']
    bg_y = params['BG_slope_y']
    bg_c = params['BG_const']
    
    xx, yy = np.meshgrid(x, y)
    
    rotxx = (xx-xc)*np.cos(th)-(yy-yc)*np.sin(th)
    rotyy = (xx-xc)*np.sin(th)+(yy-yc)*np.cos(th)
    
    model = A*np.exp(-rotxx**2/2/sx**2-rotyy**2/2/sy**2)/(2*np.pi*sx*sy)
    model = model+bg_x*xx+bg_y*yy+bg_c
    
    return model-z


def guess_RotGauss2D(x, y, z):
    '''
    Making initial guess
    x, y can be pixel number, HKL values etc.
    '''
    xx, yy = np.meshgrid(x, y)
    
    # Guessing the center
    cen = np.unravel_index(z.argmax(), z.shape)
    yc = y[cen[0]]
    xc = x[cen[1]]
    
    # Guessing the BG
    bg_c = np.nanmin(z)
    height = np.nanmax(z)-bg_c
    
    # Guessing the widths and BG slopes
    temp = z[cen[0]]
    bg_x = (temp[-1]-temp[0])/(x[-1]-x[0])
    templeft = temp[:cen[1]]
    sx_left = np.argwhere(templeft<bg_c+height/2)[-1,0]
    tempright = temp[cen[1]:]
    sx_right = np.argwhere(tempright<bg_c+height/2)[0,0]
    sx = np.abs((x[sx_right]-x[sx_left])/2.)
    
    temp = z[:, cen[1]]
    bg_y = (temp[-1]-temp[0])/(y[-1]-y[0])
    templeft = temp[:cen[0]]
    sy_left = np.argwhere(templeft<bg_c+height/2)[-1,0]
    tempright = temp[cen[0]:]
    sy_right = np.argwhere(tempright<bg_c+height/2)[0,0]
    sy = np.abs((y[sy_right]-y[sy_left])/2.)
    
    # Guessing the area
    A = 2*np.pi*sx*sy*height
    
    params = Parameters()
    params.add('Area', value=A)
    params.add('sigma_x', value=sx)
    params.add('sigma_y', value=sy)
    params.add('cen_x', value=xc)
    params.add('cen_y', value=yc)
    params.add('theta', value=1.)    # Give it a guess away from zero
    params.add('BG_slope_x', value=bg_x)
    params.add('BG_slope_y', value=bg_y)
    params.add('BG_const', value=bg_c)
    
    return params


#==============================================================================
# Anisotropic Lorentzian
#==============================================================================
def Lor2D(params, x, y, z):
    '''
    Lor
    
    Note Lor2D has a height but not an area.
    The -inf to inf integral does not converge.
    '''
    H = params['Height']
    fwhm_x = params['FWHM_x']
    fwhm_y = params['FWHM_y']
    xc = params['cen_x']
    yc = params['cen_y']
    
    bg_x = params['BG_slope_x']
    bg_y = params['BG_slope_y']
    bg_c = params['BG_const']
    
    xx, yy = np.meshgrid(x, y)
    
    model = H/(((xx-xc)*2/fwhm_x)**2+((yy-yc)*2/fwhm_y)**2+1)
    model = model+bg_x*xx+bg_y*yy+bg_c
    
    return model-z


def guess_Lor2D(x, y, z):
    '''
    Making initial guess
    x, y can be pixel number, HKL values etc.
    '''
    xx, yy = np.meshgrid(x, y)
    
    # Guessing the center
    cen = np.unravel_index(z.argmax(), z.shape)
    yc = y[cen[0]]
    xc = x[cen[1]]
    
    # Guessing the BG
    bg_c = np.nanmin(z)
    height = np.nanmax(z)-bg_c
    
    # Guessing the widths and BG slopes
    temp = z[cen[0]]
    bg_x = (temp[-1]-temp[0])/(x[-1]-x[0])
    templeft = temp[:cen[1]]
    sx_left = np.argwhere(templeft<bg_c+height/2)[-1,0]
    tempright = temp[cen[1]:]
    sx_right = np.argwhere(tempright<bg_c+height/2)[0,0]
    fwhm_x = np.abs(x[sx_right]-x[sx_left])
    
    temp = z[:, cen[1]]
    bg_y = (temp[-1]-temp[0])/(y[-1]-y[0])
    templeft = temp[:cen[0]]
    sy_left = np.argwhere(templeft<bg_c+height/2)[-1,0]
    tempright = temp[cen[0]:]
    sy_right = np.argwhere(tempright<bg_c+height/2)[0,0]
    fwhm_y = np.abs(y[sy_right]-y[sy_left])
    
    params = Parameters()
    params.add('Height', value=height)
    params.add('FWHM_x', value=fwhm_x)
    params.add('FWHM_y', value=fwhm_y)
    params.add('cen_x', value=xc)
    params.add('cen_y', value=yc)
    params.add('BG_slope_x', value=bg_x)
    params.add('BG_slope_y', value=bg_y)
    params.add('BG_const', value=bg_c)
    
    return params


#==============================================================================
# Rotated anisotropic Lorentzian
#==============================================================================
def RotLor2D(params, x, y, z):
    '''
    Rotated Lor
    
    Note Lor2D has a height but not an area.
    The -inf to inf integral does not converge.
    '''
    H = params['Height']
    fwhm_x = params['FWHM_x']
    fwhm_y = params['FWHM_y']
    xc = params['cen_x']
    yc = params['cen_y']
    th = params['theta']*np.pi/180.
    
    bg_x = params['BG_slope_x']
    bg_y = params['BG_slope_y']
    bg_c = params['BG_const']
    
    xx, yy = np.meshgrid(x, y)
    
    rotxx = (xx-xc)*np.cos(th)-(yy-yc)*np.sin(th)
    rotyy = (xx-xc)*np.sin(th)+(yy-yc)*np.cos(th)
    
    model = H/((rotxx*2/fwhm_x)**2+(rotyy*2/fwhm_y)**2+1)
    model = model+bg_x*xx+bg_y*yy+bg_c
    
    return model-z


def guess_RotLor2D(x, y, z):
    '''
    Making initial guess
    x, y can be pixel number, HKL values etc.
    '''
    xx, yy = np.meshgrid(x, y)
    
    # Guessing the center
    cen = np.unravel_index(z.argmax(), z.shape)
    yc = y[cen[0]]
    xc = x[cen[1]]
    
    # Guessing the BG
    bg_c = np.nanmin(z)
    height = np.nanmax(z)-bg_c
    
    # Guessing the widths and BG slopes
    temp = z[cen[0]]
    bg_x = (temp[-1]-temp[0])/(x[-1]-x[0])
    templeft = temp[:cen[1]]
    sx_left = np.argwhere(templeft<bg_c+height/2)[-1,0]
    tempright = temp[cen[1]:]
    sx_right = np.argwhere(tempright<bg_c+height/2)[0,0]
    fwhm_x = np.abs(x[sx_right]-x[sx_left])
    
    temp = z[:, cen[1]]
    bg_y = (temp[-1]-temp[0])/(y[-1]-y[0])
    templeft = temp[:cen[0]]
    sy_left = np.argwhere(templeft<bg_c+height/2)[-1,0]
    tempright = temp[cen[0]:]
    sy_right = np.argwhere(tempright<bg_c+height/2)[0,0]
    fwhm_y = np.abs(y[sy_right]-y[sy_left])
    
    params = Parameters()
    params.add('Height', value=height)
    params.add('FWHM_x', value=fwhm_x)
    params.add('FWHM_y', value=fwhm_y)
    params.add('cen_x', value=xc)
    params.add('cen_y', value=yc)
    params.add('theta', value=1.)    # Give it a guess away from zero
    params.add('BG_slope_x', value=bg_x)
    params.add('BG_slope_y', value=bg_y)
    params.add('BG_const', value=bg_c)
    
    return params


#==============================================================================
# Fitting funcs
# More complicated fitting funcs
# Automatic guess does not work most times
# 
# Rotated anisotropic Gaussian
#==============================================================================
def RotDoubleGauss2D(params, x, y, z):
    '''
    Rotated double Gaussian
    
    Input:
    x: x coordinate, 1D
    y: y coordinate, 1D
    z: height values
    '''
    A1 = params['Area1']
    sx1 = params['sigma1_x']
    sy1 = params['sigma1_y']
    xc1 = params['cen1_x']
    yc1 = params['cen1_y']
    th1 = params['theta1']*np.pi/180.
    
    A2 = params['Area2']
    sx2 = params['sigma2_x']
    sy2 = params['sigma2_y']
    xc2 = params['cen2_x']
    yc2 = params['cen2_y']
    th2 = params['theta2']*np.pi/180.
    
    bg_x = params['BG_slope_x']
    bg_y = params['BG_slope_y']
    bg_c = params['BG_const']
    
    xx, yy = np.meshgrid(x, y)
    
    rotxx1 = (xx-xc1)*np.cos(th1)-(yy-yc1)*np.sin(th1)
    rotyy1 = (xx-xc1)*np.sin(th1)+(yy-yc1)*np.cos(th1)
    
    rotxx2 = (xx-xc2)*np.cos(th2)-(yy-yc2)*np.sin(th2)
    rotyy2 = (xx-xc2)*np.sin(th2)+(yy-yc2)*np.cos(th2)
    
    model = A1*np.exp(-rotxx1**2/2/sx1**2-rotyy1**2/2/sy1**2)/(2*np.pi*sx1*sy1)
    model = model+A2*np.exp(-rotxx2**2/2/sx2**2-rotyy2**2/2/sy2**2)/(2*np.pi*sx2*sy2)
    model = model+bg_x*xx+bg_y*yy+bg_c
    
    return model-z


def guess_RotDoubleGauss2D(x, y, z):
    '''
    Making initial guess
    x, y can be pixel number, HKL values etc.
    
    For double Gauss, the guess would not be accurate
    '''
    xx, yy = np.meshgrid(x, y)
    
    # Guessing center 1
    cen1 = np.unravel_index(z.argmax(), z.shape)
    yc1 = y[cen1[0]]
    xc1 = x[cen1[1]]
    
    # Guessing the BG
    bg_c = np.nanmin(z)
    height1 = np.nanmax(z)-bg_c
    
    # Guessing the widths and BG slopes
    temp = z[cen1[0]]
    bg_x = (temp[-1]-temp[0])/(x[-1]-x[0])
    templeft = temp[:cen1[1]]
    sx_left = np.argwhere(templeft<bg_c+height1/2)[-1,0]
    tempright = temp[cen1[1]:]
    sx_right = np.argwhere(tempright<bg_c+height1/2)[0,0]
    sx1 = np.abs((x[sx_right]-x[sx_left])/2.)
    
    temp = z[:, cen1[1]]
    bg_y = (temp[-1]-temp[0])/(y[-1]-y[0])
    templeft = temp[:cen1[0]]
    sy_left = np.argwhere(templeft<bg_c+height1/2)[-1,0]
    tempright = temp[cen1[0]:]
    sy_right = np.argwhere(tempright<bg_c+height1/2)[0,0]
    sy1 = np.abs((y[sy_right]-y[sy_left])/2.)
    
    # Guessing the area
    A1 = 2*np.pi*sx1*sy1*height1
    
    params = Parameters()
    params.add('Area1', value=A1)
    params.add('sigma1_x', value=sx1)
    params.add('sigma1_y', value=sy1)
    params.add('cen1_x', value=xc1)
    params.add('cen1_y', value=yc1)
    params.add('theta1', value=1.)    # Give it a guess away from zero
    
    # Not a good guess
    params.add('Area2', value=A1)
    params.add('sigma2_x', value=sx1)
    params.add('sigma2_y', value=sy1)
    params.add('cen2_x', value=-xc1)
    params.add('cen2_y', value=-yc1)
    params.add('theta2', value=-1.)    # Give it a guess away from zero
    
    params.add('BG_slope_x', value=bg_x)
    params.add('BG_slope_y', value=bg_y)
    params.add('BG_const', value=bg_c)
    
    return params


#==============================================================================
# Rotated double anisotropic Lorentzian
#==============================================================================
def RotDoubleLor2D(params, x, y, z):
    '''
    Rotated double Lor
    
    Note Lor2D has a height but not an area.
    The -inf to inf integral does not converge.
    '''
    H1 = params['Height1']
    fwhm1_x = params['FWHM1_x']
    fwhm1_y = params['FWHM1_y']
    xc1 = params['cen1_x']
    yc1 = params['cen1_y']
    th1 = params['theta1']*np.pi/180.
    
    H2 = params['Height2']
    fwhm2_x = params['FWHM2_x']
    fwhm2_y = params['FWHM2_y']
    xc2 = params['cen2_x']
    yc2 = params['cen2_y']
    th2 = params['theta2']*np.pi/180.
    
    bg_x = params['BG_slope_x']
    bg_y = params['BG_slope_y']
    bg_c = params['BG_const']
    
    xx, yy = np.meshgrid(x, y)
    
    rotxx1 = (xx-xc1)*np.cos(th1)-(yy-yc1)*np.sin(th1)
    rotyy1 = (xx-xc1)*np.sin(th1)+(yy-yc1)*np.cos(th1)
    
    rotxx2 = (xx-xc2)*np.cos(th2)-(yy-yc2)*np.sin(th2)
    rotyy2 = (xx-xc2)*np.sin(th2)+(yy-yc2)*np.cos(th2)
    
    model = H1/((rotxx1*2/fwhm1_x)**2+(rotyy1*2/fwhm1_y)**2+1)
    model = model+H2/((rotxx2*2/fwhm2_x)**2+(rotyy2*2/fwhm2_y)**2+1)
    
    model = model+bg_x*xx+bg_y*yy+bg_c
    
    return model-z


def guess_RotDoubleLor2D(x, y, z):
    '''
    Making initial guess
    x, y can be pixel number, HKL values etc.
    
    For double Lor, the guess would not be accurate
    '''
    xx, yy = np.meshgrid(x, y)
    
    # Guessing center 1
    cen1 = np.unravel_index(z.argmax(), z.shape)
    yc1 = y[cen1[0]]
    xc1 = x[cen1[1]]
    
    # Guessing the BG
    bg_c = np.nanmin(z)
    height1 = np.nanmax(z)-bg_c
    
    # Guessing the widths and BG slopes
    temp = z[cen1[0]]
    bg_x = (temp[-1]-temp[0])/(x[-1]-x[0])
    templeft = temp[:cen1[1]]
    sx_left = np.argwhere(templeft<bg_c+height1/2)[-1,0]
    tempright = temp[cen1[1]:]
    sx_right = np.argwhere(tempright<bg_c+height1/2)[0,0]
    fwhm1_x = np.abs(x[sx_right]-x[sx_left])
    
    temp = z[:, cen1[1]]
    bg_y = (temp[-1]-temp[0])/(y[-1]-y[0])
    templeft = temp[:cen1[0]]
    sy_left = np.argwhere(templeft<bg_c+height1/2)[-1,0]
    tempright = temp[cen1[0]:]
    sy_right = np.argwhere(tempright<bg_c+height1/2)[0,0]
    fwhm1_y = np.abs(y[sy_right]-y[sy_left])
    
    params = Parameters()
    params.add('Height1', value=height1)
    params.add('FWHM1_x', value=fwhm1_x)
    params.add('FWHM1_y', value=fwhm1_y)
    params.add('cen1_x', value=xc1)
    params.add('cen1_y', value=yc1)
    params.add('theta1', value=1.)    # Give it a guess away from zero
    
    # Not a good guess
    params.add('Height2', value=height1)
    params.add('FWHM2_x', value=fwhm1_x)
    params.add('FWHM2_y', value=fwhm1_y)
    params.add('cen2_x', value=-xc1)
    params.add('cen2_y', value=-yc1)
    params.add('theta2', value=-1.)    # Give it a guess away from zero
    
    params.add('BG_slope_x', value=bg_x)
    params.add('BG_slope_y', value=bg_y)
    params.add('BG_const', value=bg_c)
    
    return params