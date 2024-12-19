# importing system packages
import os
import sys
import glob
import h5py
import time
import itertools
from tqdm import tqdm, trange

# importing the workhorse
import numpy as np
import pandas as pd
from scipy import io, signal, interpolate, ndimage,sparse

# tiff packages
import tifffile

# import all fitting packages
from lmfit import Minimizer, Parameters, report_fit

from visReduce import *
from XPCSana import *
from Hao_XPCS_utils import *
import shot_object_v2 as ss
from single_shot_helper_functions import *
from Hao_speckle_utils_v2_2 import *

# smdPath
# smdDir = '/cds/data/drpsrcf/xcs/xcsly3520/scratch/hdf5/smalldata'
smdDir='/cds/data/psdm/xcs/xcsly3520/hdf5/smalldata'

p = {
    'crlx': 'epicsAll/crl_Be_xpos',
    'crly': 'epicsAll/crl_Be_ypos',
    'crlz': 'epicsAll/crl_Be_zpos',
    'samx': 'epicsAll/sam_x',
    'samy': 'epicsAll/sam_y',
    'i0' : 'ipm4/sum',
    'delays_raw': 'epicsAll/las_Target_Time_ns',
    'laser' : 'lightStatus/laser',
    'xray' : 'lightStatus/xray',
    'on': 'evr/code_87',
    'before': 'evr/code_190',
    'after': 'evr/code_193',
    'et': 'event_time',
    'epix1_ROI_0' : 'epix_1/ROI_0_area',
    'epix2_ROI_0' : 'epix_2/ROI_0_area',
    'epix3_ROI_0' : 'epix_3/ROI_0_area',
    'epix4_ROI_0' : 'epix_4/ROI_0_area',
    'epix1_ROI_sum' : 'epix_1/ROI_0_sum',
    'epix2_ROI_sum' : 'epix_2/ROI_0_sum',
    'epix3_ROI_sum' : 'epix_3/ROI_0_sum',
    'epix4_ROI_sum' : 'epix_4/ROI_0_sum'
}
def cal_delay(l):
    return np.around((44694.115877 - l)/10)*10 # ns

def coord_fiduc(xs,ys):
    if not len(xs)==len(ys):
        return None
    result = np.zeros(xs.shape)
    quad_dict={3:.1,1:.2,-3:.3,-1:.4}
    xsr=np.around(xs,4)
    ysr=np.around(ys,4)
    for ind,(x,y) in enumerate(zip(xsr,ysr)):
        result[ind]=quad_dict[np.sign(x)+2*np.sign(y)]+np.abs(x)*1e-2*10**-np.floor(np.log10(np.abs(x)))+np.abs(y)*1e-7*10**-np.floor(np.log10(np.abs(y)))
#         print(ind,(x,y),result[ind],quad_dict[np.sign(x)+2*np.sign(y)]+np.abs(x)*1e-2*10**-np.floor(np.log10(np.abs(x)))+np.abs(y)*1e-7*10**-np.floor(np.log10(np.abs(y))))
    return result

def read_fiduc(fid):
    inverse_quad_dict={1:[1,1],2:[-1,1],3:[-1,-1],4:[1,-1]}
    xs=np.zeros_like(fid)
    ys=np.zeros_like(fid)
    for ind,f in enumerate(fid):
        q=int(f*10)
        xs[ind]=np.floor(f*1e2-q*1e1)*inverse_quad_dict[q][0]
        ys[ind]=(f*1e7-q*1e6-xs[ind]*1e5)*inverse_quad_dict[q][1]
    return xs,ys

def save_TT_tiff(run,TT,str_ids,delay,fiducs,i0_thres,pix_thres,ROI_size,ROI_list,norm_is_before,norm_is_on,norm_is_after,weights=None):
    saveloc='SS_data/run{}/TT/Gorfmann-Q-Sweep/{}'.format(str(run),str_ids)
    if not os.path.isdir(saveloc):
        os.makedirs(saveloc)
    tifffile.imwrite(os.path.join(saveloc,'TTs_{}.tiff'.format(str(int(delay)))),TT)
    if not weights is None:
        tifffile.imwrite(os.path.join(saveloc,'weights_{}.tiff'.format(str(int(delay)))),weights)
    np.savez(os.path.join(saveloc,'ROI_fiducs_{}'.format(str(int(delay)))),ROI=ROI_list,fiducs=fiducs,norm_is_before=norm_is_before,norm_is_on=norm_is_on,norm_is_after=norm_is_after)
    
def plot_triplet(before,on,after,i0s,titleStr=None):
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    ax[0].imshow(np.nanmean(normalize(before,i0s[:before.shape[0]]),axis=0))
    try:
        ax[1].imshow(on[0]/i0s[before.shape[0]])
    except:
        ax[1].imshow(np.zeros_like(on[0]))
    ax[2].imshow(np.nanmean(normalize(after,i0s[-after.shape[0]:]),axis=0))
    if not titleStr is None:ax[1].set_title(titleStr)
        

if __name__ == "__main__":
    runID = int(sys.argv[1])
    detector =int(sys.argv[2])
    #shotsperdelay=18
    fpath = os.path.join(smdDir, 'xcsly3520_Run{:04d}.h5'.format(runID))
    h = h5py.File(fpath, 'r')
    
    delays=cal_delay(h[p['delays_raw']][:])  #units=ns
    uniq_delays=np.unique(delays)
    
    on_raw=np.array(h[p['on']][:],dtype=bool)
    after_raw=np.array(h[p['after']][:],dtype=bool)
    before_raw=np.array(h[p['before']][:],dtype=bool)
    
    on=np.zeros_like(on_raw,dtype=bool)
    true_ons=np.where(on_raw)[0]+1
    on[true_ons]=True

    before=before_raw.copy()
    before[true_ons-1]=True

    after=after_raw.copy()
    after[true_ons]=False
    
    laser=np.array(h[p['laser']][:],dtype=bool)
    xray=np.array(h[p['xray']][:],dtype=bool)
    
    i0=h[p['i0']][:]

    ys=h[p['samy']][:]
    # uniq_ys=np.unique(ys)

    xs=h[p['samx']][:]
    # uniq_xs=np.unique(xs)

    fiducs=coord_fiduc(xs,ys)
    uniq_fiducs=np.unique(fiducs)
    
    if detector==1:
        epix=h[p['epix1_ROI_0']]
        sums=h[p['epix1_ROI_sum']]
    elif detector==2:
        epix=h[p['epix2_ROI_0']]
        sums=h[p['epix2_ROI_sum']]
    elif detector==3:
        epix=h[p['epix3_ROI_0']]
        sums=h[p['epix3_ROI_sum']]
    elif detector==4:
        epix=h[p['epix4_ROI_0']]
        sums=h[p['epix4_ROI_sum']]
    else:
        print('invalid detector num!')
        sys.exit()
    pix_thres=44
    i0_thres=100
    ROI_size=50
    
    i0_thres=100
    i0[i0<=i0_thres]=np.nan

    mask=np.ones((epix.shape[1],epix.shape[2]),dtype=bool)
    #Run 26-Run 29
    if detector==3:
        mask[351,:]=False
        mask[352,:]=False
        mask[:,83]=False
        mask[:,84]=False
        mask[:351,68:83]=False
    elif detector==1:
        mask[351,:]=False
        mask[352,:]=False
        mask[:,382]=False
        mask[:,383]=False
        mask[:352,382:480]=False
    delta_xs=[0]#np.arange(-100,101,100)
    delta_ys=[0]#np.arange(-100,101,100)
    delta_ROIs=np.array([[dx,dx,dy,dy] for dy in delta_ys for dx in delta_xs], dtype=int)
    for dR in delta_ROIs:
        agreg_data=dict.fromkeys(uniq_delays)
        print('(dQx,dQy)=',(dR[0],dR[-1]))
        for delay in uniq_delays:
            agreg_data[delay]={'TT':[],'ROI':[],'fiducs':[],'norm_is_before':[],'norm_is_on':[],'norm_is_after':[],'weights':[]}

        for ind,fiduc in enumerate(uniq_fiducs):
            shots=fiducs==fiduc
        #     print(on[shots])
            on_in_shots=np.logical_and(shots,on)
            before_in_shots=np.logical_and(shots,before)
            after_in_shots=np.logical_and(shots,after)

            #abort filters
            c0=on_in_shots.sum()==1
            c1=np.all(delays[shots]==delays[shots][0])
            c2=shots.sum()==29  #change this if shape changes
        #     c3=np.all(np.invert(np.isnan(i0[on_in_shots])))
            c4=not np.all(np.isnan(i0[shots]))
        #     print([c0,c1,c2,c3])
            if not all([c0,c1,c2,c4]):
                print('abort mask failed for shots ', min(np.where(shots)[0]), ' to ', max(np.where(shots)[0]))
                continue

            #check before shots for previous conversion
            before_sum=np.nanmean(sums[before_in_shots]/i0[before_in_shots])
            if before_sum>1000: #determined empirically using run 232
        #         displayImg(np.nanmean(normalize(epix3[np.where(before_in_shots)[0],:,:],i0[before_in_shots]),axis=0),titleStr=str(before_sum))
                print('shots ',min(np.where(shots)[0]), ' to ', max(np.where(shots)[0]), ' were converted before pump. Delay = ',delays[on_in_shots])
                continue

            after_sum=np.nanmean(sums[after_in_shots]/i0[after_in_shots])
            if after_sum<2500:  #determined emperically using run 234
        #         displayImg(np.nanmean(normalize(epix3[np.where(after_in_shots)[0],:,:],i0[after_in_shots]),axis=0),titleStr=str(after_sum))
                print('shots ',
                      min(np.where(shots)[0]),
                      ' to ', max(np.where(shots)[0]),
                      ' didn\'t convert. Delay = ',delays[on_in_shots])
                continue
            #convert bad i0 shots to NaN epix3
            shot_before=epix[np.where(before_in_shots)[0],:,:]
            shot_on=epix[np.where(on_in_shots)[0],:,:]
            shot_after=epix[np.where(after_in_shots)[0],:,:]

            agreg_data[delays[on_in_shots][0]]['norm_is_before'].append(np.sum(np.array([*shot_before*mask]),axis=(1,2))/i0[before_in_shots])
            agreg_data[delays[on_in_shots][0]]['norm_is_on'].append(np.sum(np.array([*shot_on*mask]),axis=(1,2))/i0[on_in_shots])
            agreg_data[delays[on_in_shots][0]]['norm_is_after'].append(np.sum(np.array([*shot_after*mask]),axis=(1,2))/i0[after_in_shots])

            shot_before[shot_before<pix_thres]=0.0
            shot_on[shot_on<pix_thres]=0.0
            shot_after[shot_after<pix_thres]=0.0

            com_row,com_col,c2r,c2c=findAfterPeakCenter1D(shot_after,i0[after_in_shots],peak_mask=np.invert(mask),seePlots=False)


            ROI=np.array([com_col-ROI_size//2,com_col+ROI_size//2,com_row-ROI_size//2,com_row+ROI_size//2],dtype=int)+dR
            if ROI[0]<0 or ROI[1]>shot_on.shape[2] or ROI[2]<0 or ROI[3]>shot_on.shape[1]:
                continue
        #     plotTwoTime(TT,vmin=1,vmax=1.4)
            agreg_data[delays[on_in_shots][0]]['TT'].append(spotTT(shot_before,shot_on,shot_after,ROI,i0s=i0[shots],mode='Gorfmann'))
            agreg_data[delays[on_in_shots][0]]['ROI'].append(ROI)
            agreg_data[delays[on_in_shots][0]]['fiducs'].append(fiduc)
            agreg_data[delays[on_in_shots][0]]['weights'].append(np.outer(i0[shots],i0[shots]))
            print('shots ',min(np.where(shots)[0]), ' to ', max(np.where(shots)[0]), ' were fine. Delay = ',delays[on_in_shots])

        for delay in uniq_delays:
            print('No. of spots saved at delay',delay,'ns is',len(agreg_data[delay]['TT']))
            if len(agreg_data[delay]['TT'])==0:
                print('no spots at above delay!')
                continue
            save_TT_tiff(runID,
                         np.array(agreg_data[delay]['TT']),
                         'epix{}_i0-{}_pix-{}_ROI-{}_qxoff{}_qyoff{}_no-photonization'\
                         .format(detector,i0_thres,pix_thres,ROI_size,dR[0],dR[-1]),
                         delay,
                         np.array(agreg_data[delay]['fiducs']),
                         i0_thres,
                         pix_thres,
                         ROI_size,
                         np.array(agreg_data[delay]['ROI']),
                         np.array(agreg_data[delay]['norm_is_before']),
                         np.array(agreg_data[delay]['norm_is_on']),
                         np.array(agreg_data[delay]['norm_is_after']),
                         np.array(agreg_data[delay]['weights'])
                        )
    h.close()
