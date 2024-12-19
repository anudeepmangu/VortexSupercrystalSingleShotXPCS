#==============================================================================
# XPCS speckle analysis
# 2020-06-02
# 
# Ver. 4
# Yue Cao (ycao@colorado.edu)
#
#==============================================================================

# import all plotting packages
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
from scipy import io, signal, interpolate, stats
from scipy.special import gamma

# from pyFAI import *

# tiff packages
import tifffile

from lmfit import Minimizer, Parameters, report_fit, Model

#==============================================================================
# XPCS Utils - generating waterfall from a ROI
#==============================================================================
def genLineROI(speckle, cen_pix=[], cut='H'):
    '''
    speckle in 3D:
    0d: time
    1d: y
    2d: x
    
    cut: can be 'H' or 'V'
    '''
    if cen_pix==[]:
        ycen = (speckle.shape[1]-1)//2
        xcen = (speckle.shape[2]-1)//2
    else:
        xcen = cen_pix[0]
        ycen = cen_pix[1]

    if cut=='H':
        waterfall = speckle[:, ycen, :]
    if cut=='V':
        waterfall = speckle[:, :, xCen]

    return waterfall


def genBoxROI(speckle, ROI):
    '''
    speckle in 3D:
    0d: time
    1d: y
    2d: x
    
    ROI = [xmin, xmax, ymin, ymax]
    '''
    temp = speckle[:, ROI[2]:ROI[3], ROI[0]:ROI[1]]
    waterfall = temp.reshape(temp.shape[0], temp.shape[1]*temp.shape[2])

    return waterfall


def genCirROI(speckle, rmin, rmax, cen_pix=[]):
    '''
    speckle in 3D:
    0d: time
    1d: y
    2d: x
    '''
    if cen_pix==[]:
        ycen = (speckle.shape[1]-1)/2
        xcen = (speckle.shape[2]-1)/2
    else:
        xcen = cen_pix[0]
        ycen = cen_pix[1]

    yline = np.arange(speckle.shape[1])-ycen
    xline = np.arange(speckle.shape[2])-xcen
    xmesh, ymesh = np.meshgrid(xline, yline)
    radius = xmesh**2+ymesh**2

    mask=np.logical_and(radius>rmin**2,radius<rmax**2)

    # Apply masks and generate waterfall
    waterfall = []
    for i in range(speckle.shape[0]):
        temp = speckle[i][mask]
        # print(np.size(temp))
        waterfall.append(temp)

    waterfall = np.array(waterfall)

    return waterfall


#==============================================================================
# XPCS Utils - calculating g2mat and g2
#==============================================================================

# def _calc_corr(im1, im2):
#     ab = np.nanmean(np.multiply(im1, im2))
#     aa = np.nanmean(im1)
#     bb = np.nanmean(im2)
    
#     if aa*bb==0:
#         corr = np.nan
#     else:
#         corr = ab/aa/bb
#     return corr


# def _gen_twotime(waterfall):
#     '''
#     Calc two-time for a stack of images or lines
#     Slower ver
#     0d: Time
#     '''
#     g2mat = np.zeros((waterfall.shape[0], waterfall.shape[0]))
    
#     for i in range(waterfall.shape[0]):
#         for j in range(i, waterfall.shape[0]):
#             g2mat[i, j] = _calc_corr(waterfall[i], waterfall[j])
#             g2mat[j, i] = g2mat[i, j]
#     return g2mat


def calcTwoTime(waterfall, mode='Sutton',timing=True):
    '''
    Calc two-time for a stack of images or lines
    Quick ver, n**2/2 complexity
    0d: Time
    
    Mode:
    S:    Following M. Sutton, M. Opt. Express, 11, 2268–2277 (2003)
    
    '''
    ts = time.time()
    
    if mode=='Sutton':
#         print(waterfall.dtype)
        ab = np.zeros((waterfall.shape[0], waterfall.shape[0]))
        aa = np.zeros(waterfall.shape[0])

        for i in range(waterfall.shape[0]):
            for j in range(i, waterfall.shape[0]):
                ab[i, j] = np.nanmean(np.multiply(waterfall[i], waterfall[j]))
                ab[j, i] = ab[i, j]
            aa[i] = np.nanmean(waterfall[i])

        g2mat = np.divide(ab, np.outer(aa, aa))
    elif mode=='SuttonStd':
        ab = np.zeros((waterfall.shape[0], waterfall.shape[0]))
        aa_m = np.zeros(waterfall.shape[0])
        aa_s = np.zeros(waterfall.shape[0])

        for i in range(waterfall.shape[0]):
            for j in range(i, waterfall.shape[0]):
                ab[i, j] = np.nanmean(np.multiply(waterfall[i], waterfall[j]))
                ab[j, i] = ab[i, j]
            aa_m[i] = np.nanmean(waterfall[i])
            aa_s[i] = np.nanstd(waterfall[i])
        
        numerator=ab-np.outer(aa_m,aa_m)
        denominator=np.outer(aa_s,aa_s)
        g2mat = np.divide(numerator,denominator)
    elif mode == 'vector':
        wf_mean = wf.nanmean(axis=1).reshape(-1, 1)
        cov = (wf @ wf.T) / wf.shape[1]
        g2mat = cov / (wf_mean @ wf_mean.T)
    elif mode== 'gorfmann' or mode=='Gorfmann':
        subt_waterfall=np.array([waterfall[k]-np.mean(waterfall[k]) for k in range(waterfall.shape[0])])
        g2mat=np.zeros((waterfall.shape[0], waterfall.shape[0]))
        for i in range(waterfall.shape[0]):
            for j in range(i, waterfall.shape[0]):
                numerator = np.nansum(np.multiply(subt_waterfall[i], subt_waterfall[j]))
                denominator=np.sqrt(np.nansum(subt_waterfall[i]**2)*np.nansum(subt_waterfall[j]**2))
                g2mat[i,j]=numerator/denominator
                g2mat[j,i]=g2mat[i,j]
               
    else:
        g2mat=None
        
        
    te = time.time()
    if timing:
        print('***** XPCS two-time processed in {} sec *****'.format(te-ts))
    
    return g2mat

def calcOneLineTT(waterfall, reference=-1, mode='Sutton',timing=True):
    '''
    Calc one line of TT without calculating the full TT
    
    Mode:
    S:    Following M. Sutton, M. Opt. Express, 11, 2268–2277 (2003)
    
    '''
    ts = time.time()
    
    if mode=='Sutton':
#         print(waterfall.dtype)
#         C=np.array([np.nanmean(np.multiply(waterfall[i], waterfall[reference]))/(np.nanmean(waterfall[i])*np.nanmean(waterfall[reference])) for i in range(waterfall.shape[0])])
        C=np.mean(waterfall*waterfall[reference],axis=1)/(np.mean(waterfall,axis=1)*np.mean(waterfall[reference]))
    
    elif mode=='SuttonStd':
        C=np.array([(np.nanmean(np.multiply(waterfall[i], waterfall[reference]))-(np.nanmean(waterfall[i])*np.nanmean(waterfall[reference])))/(np.nanstd(waterfall[i])*np.nanstd(waterfall[reference])) for i in range(waterfall.shape[0])])

    elif mode == 'vector':
        wf_mean = np.nanmean(waterfall,axis=1).reshape(-1, 1)
        cov = (waterfall @ waterfall.T) / waterfall.shape[1]
        g2mat = cov / (wf_mean @ wf_mean.T)
        C=g2mat[reference]
    elif mode== 'gorfmann' or mode=='Gorfmann':
        subt_waterfall=np.array([waterfall[k]-np.mean(waterfall[k]) for k in range(waterfall.shape[0])])
        C=np.array([np.nansum(np.multiply(subt_waterfall[i], subt_waterfall[reference]))/(np.nansum(subt_waterfall[i]**2)*np.nansum(waterfall[reference]**2)) for i in range(waterfall.shape[0])])
               
    else:
        C=None
        
        
    te = time.time()
    if timing:
        print('***** XPCS two-time processed in {} sec *****'.format(te-ts))
    
    return C


def calcG2(g2mat, mask=[]):
    '''
    mask has to have the same shape (N by N) as g2mat.
    Use mask for filtering out unwanted contributions to g2.
    '''
    # Generate g2 from two-time
    if not mask==[]:
        g2mat = np.multiply(g2mat, mask)
    
    g2 = []
    sigma_g2 = []
    for i in range(g2mat.shape[0]):
        temp = np.nanmean(np.diagonal(g2mat, i))
        g2.append(temp)
        
        temp = np.nanstd(np.diagonal(g2mat, i))
        sigma_g2.append(temp)
    
    g2 = np.array(g2)
    sigma_g2 = np.array(sigma_g2)
    
    return g2, sigma_g2


def plotTwoTime(g2mat, vmin, vmax):
    '''
    Plotting two time
    '''
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    im = ax.imshow(g2mat, cmap=cm.rainbow, vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    plt.tight_layout()
    
    return


#==============================================================================
# Fits
#==============================================================================

        
def GammaDis(param, x, y):
    '''
    Gamma distribution
    x: I/<I>
    y: normalized PDF
    '''
    M = 1./param['beta']
    A = param['Amp']
    # xavg = np.nanmean(x)
    # model = A*(M/xavg)**M*x**(M-1)*np.exp(-M*x/xavg)/gamma(M)
    model = A*x**(M-1)*np.exp(-M*x)
    return model-y


def fit_Gamma(pdf, bincen, verbose=False, plotFit=False):
    '''
    Inputs:
    pdf:        Probability distribution function
    bincen:     Centers of the bins
    '''
    
    params = Parameters()
    params.add('beta', value=0.5)
    params.add('Amp', value=1.)

    minner = Minimizer(GammaDis, params, fcn_args=(bincen, pdf))
    result = minner.minimize()
    fitted = pdf + result.residual
    
    chisqr = result.redchi
    beta = result.params['beta'].value
    beta_err = result.params['beta'].stderr

    if verbose:
        report_fit(result)
        
    if plotFit:
        plt.figure()
        plt.plot(bincen, pdf)
        plt.plot(bincen, fitted)
    
    return fitted, chisqr, beta, beta_err


# def SimpleExp(params, x, y):
#     tau = params['tau']
    
#     model = 1.+np.exp(-x/tau)**2
#     return model-y


# def fit_SimpleExp(x, y, verbose=False, plotFit=False):
#     params = Parameters()
#     params.add('tau', value=10.)
    
#     minner = Minimizer(SimpleExp, params, fcn_args=(x, y))
#     result = minner.minimize()
#     fitted = y + result.residual
    
#     chisqr = result.redchi
#     tau = result.params['tau'].value
#     tau_err = result.params['tau'].stderr

#     if verbose:
#         report_fit(result)
        
#     if plotFit:
#         plt.figure()
#         plt.plot(x, y)
#         plt.plot(x, fitted)
    
#     return fitted, chisqr, tau, tau_err


# def ExpDecay(params, x, y):
#     b = params['beta']
#     tau = params['tau']
#     b0 = params['beta_0']
    
#     model = b0+b*np.exp(-x/tau)**2
#     return model-y


# def fit_ExpDecay(x, y, verbose=False, plotFit=False):
#     params = Parameters()
#     params.add('beta', value=0.5)
#     params.add('tau', value=10.)
#     params.add('beta_0', value=1.)
    
#     minner = Minimizer(ExpDecay, params, fcn_args=(x, y))
#     result = minner.minimize()
#     fitted = y + result.residual
    
#     chisqr = result.redchi
#     tau = result.params['tau'].value
#     tau_err = result.params['tau'].stderr

#     if verbose:
#         report_fit(result)
        
#     if plotFit:
#         plt.figure()
#         plt.plot(x, y)
#         plt.plot(x, fitted) 
    
#     return fitted, chisqr, tau, tau_err



def StretchDecay(params, x, y):
    b = params['beta']
    tau = params['tau']
    s = params['stretch'] # stretching exponent
    b0 = params['beta_0']
    
    model = b0+b*np.exp(-(x/tau)**s)**2
    return model-y


def fit_StretchDecay(x, y, init_params=(0.5,10.,1.,1.),verbose=False, plotFit=False):
    params = Parameters()
    params.add('beta', value=init_params[0])
    params.add('tau', value=init_params[1])
    params.add('stretch', value=init_params[2])
    params.add('beta_0', value=init_params[3])
    
    minner = Minimizer(StretchDecay, params, fcn_args=(x, y))
    result = minner.minimize()
    fitted = y + result.residual
    
    chisqr = result.redchi
    tau = result.params['tau'].value
    tau_err = result.params['tau'].stderr
    s = result.params['stretch'].value
    s_err = result.params['stretch'].stderr

    if verbose:
        report_fit(result)
        
    if plotFit:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, fitted) 
    
    return fitted, chisqr, tau, tau_err, s, s_err

def StretchDecayNoAmpChange(params, x, y):
    tau = params['tau']
    s = params['stretch'] # stretching exponent
    
    model = np.exp(-(x/tau)**s)
    return model-y


def fit_StretchDecayNoAmpChange(x, y, init_params=(10.,1.),verbose=False, plotFit=False):
    params = Parameters()
    params.add('tau', value=init_params[0],min=0)
    params.add('stretch', value=init_params[1],min=0)
    
    minner = Minimizer(StretchDecayNoAmpChange, params, fcn_args=(x, y))
    result = minner.minimize()
    fitted = y + result.residual
    
    chisqr = result.redchi
    tau = result.params['tau'].value
    tau_err = result.params['tau'].stderr
    s = result.params['stretch'].value
    s_err = result.params['stretch'].stderr

    if verbose:
        report_fit(result)
        
    if plotFit:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, fitted) 
    
    return fitted, chisqr, tau, tau_err, s, s_err

def StretchDecayAmpChange(params, x, y):
    A = params['amplitude']
    tau = params['tau']
    s = params['stretch'] # stretching exponent
    
    model = A*np.exp(-(x/tau)**s)
    return model-y


def fit_StretchDecayAmpChange(x, y, init_params=(1.,10.,1.),verbose=False, plotFit=False):
    params = Parameters()
    params.add('amplitude', value=init_params[0],min=0)
    params.add('tau', value=init_params[1],min=0)
    params.add('stretch', value=init_params[2],min=0)
    
    minner = Minimizer(StretchDecayAmpChange, params, fcn_args=(x, y))
    result = minner.minimize()
    fitted = y + result.residual
    
    chisqr = result.redchi
    A=result.params['amplitude'].value
    A_err=result.params['amplitude'].stderr
    tau = result.params['tau'].value
    tau_err = result.params['tau'].stderr
    s = result.params['stretch'].value
    s_err = result.params['stretch'].stderr

    if verbose:
        report_fit(result)
        
    if plotFit:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, fitted) 
    
    return fitted, chisqr, A,A_err,tau, tau_err, s, s_err


def StretchAsympRise(params, x, y):
    A = params['amplitude']
    C_inf=params['asymptote']
    tau = params['tau']
    s = params['stretch'] # stretching exponent
    
    model = C_inf - A*np.exp(-(x/tau)**s)
    return model-y


def fit_StretchAsympRise(x, y, init_params=(1.,1.15,10.,1.),mins=(0,0,0,0),maxs=(np.inf,np.inf,np.inf,np.inf),verbose=False, plotFit=False):
    params = Parameters()
    params.add('amplitude', value=init_params[0],min=mins[0],max=maxs[0])
    params.add('asymptote', value=init_params[1],min=mins[1],max=maxs[1])
    params.add('tau', value=init_params[2],min=mins[2],max=maxs[2])
    params.add('stretch', value=init_params[3],min=mins[3],max=maxs[3])
    
    minner = Minimizer(StretchAsympRise, params, fcn_args=(x, y))
    result = minner.minimize()
    fitted = y + result.residual
    
    chisqr = result.redchi
    A=result.params['amplitude'].value
    A_err=result.params['amplitude'].stderr
    C_inf=result.params['asymptote'].value
    C_inf_err=result.params['asymptote'].stderr
    tau = result.params['tau'].value
    tau_err = result.params['tau'].stderr
    s = result.params['stretch'].value
    s_err = result.params['stretch'].stderr

    if verbose:
        report_fit(result)
        
    if plotFit:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, fitted) 
    
    return fitted, chisqr, A,A_err,C_inf,C_inf_err,tau, tau_err, s, s_err

def Sum2StretchDecayNoBase(params, x, y):
    tau1 = params['tau1']
    s1 = params['stretch1'] # stretching exponent
    tau2 = params['tau2']
    s2 = params['stretch2'] # stretching exponent
    w=params['weight']
#     print(params)
    model = w*np.exp(-(x/tau1)**s1)+(1-w)*np.exp(-(x/tau2)**s2)
    return model-y


def fit_Sum2StretchDecayNoBase(x, y, init_params=(10.,1.,10.,1.,0.5),verbose=False, plotFit=False):
    params = Parameters()
    params.add('tau1', value=init_params[0],min=0)
    params.add('stretch1', value=init_params[1],min=0)
    params.add('tau2', value=init_params[2],min=0)
    params.add('stretch2', value=init_params[3],min=0)
    params.add('weight', value=init_params[4],min=0,max=1)
    
    minner = Minimizer(Sum2StretchDecayNoBase, params, fcn_args=(x, y))
    result = minner.minimize()
    fitted = y + result.residual
    
    chisqr = result.redchi
    tau1 = result.params['tau1'].value
    tau1_err = result.params['tau1'].stderr
    s1 = result.params['stretch1'].value
    s1_err = result.params['stretch1'].stderr
    
    tau2 = result.params['tau2'].value
    tau2_err = result.params['tau2'].stderr
    s2 = result.params['stretch2'].value
    s2_err = result.params['stretch2'].stderr
    
    w=result.params['weight'].value
    w_err=result.params['weight'].stderr

    if verbose:
        report_fit(result)
        
    if plotFit:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, fitted) 
    
    return fitted, chisqr, (tau1,tau2), (tau1_err,tau2_err), (s1,s2), (s1_err,s2_err),w,w_err


# if sigma is needed

def SimpleExp(x, tau):
    b0 = 1.
    b = 1.
    return b0+b*np.exp(-x/tau)**2


def fit_SimpleExp(x, y, tau_ini=10.,weights=None, verbose=False, plotFit=False):
    expmodel = Model(SimpleExp)
    result = expmodel.fit(y, x=x, tau=tau_ini, weights=weights)

    # When weights is used, the following is no longer true
    # fitted = y + result.residual
    
    chisqr = result.redchi
    tau = result.params['tau'].value
    tau_err = result.params['tau'].stderr
    
    fitted = SimpleExp(x, tau)

    if verbose:
        report_fit(result)
        
    if plotFit:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, fitted) 
    
    return fitted, chisqr, tau, tau_err

def ExpDecay(x, b, tau, b0):
    return b0+b*np.exp(-x/tau)**2


def fit_ExpDecay(x, y, weights=None, verbose=False, plotFit=False):
    expmodel = Model(ExpDecay)
    result = expmodel.fit(y, x=x, b=0.5, tau=10., b0=1., weights=weights)

    # When weights is used, the following is no longer true
    # fitted = y + result.residual
    
    chisqr = result.redchi
    tau = result.params['tau'].value
    tau_err = result.params['tau'].stderr
    b = result.params['b'].value
    b0 = result.params['b0'].value
    
    fitted = ExpDecay(x, b, tau, b0)

    if verbose:
        report_fit(result)
        
    if plotFit:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, fitted) 
    
    return fitted, chisqr, tau, tau_err
