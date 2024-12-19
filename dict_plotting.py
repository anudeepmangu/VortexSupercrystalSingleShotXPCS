import numpy as np
import matplotlib.pyplot as plt
from visReduce import *
from XPCSana import *
from Hao_XPCS_utils import *

def plotDict(d,xlab='',ylab='',titleStr='',style='',saveLoc=''):
    #x=[]
    #y=[]
    #for key in d.keys():
    #    x.append(key)
    #    y.append(d[key])
    #x=np.array(x)
    #y=np.array(y)
    d_items=sorted(d.items())
    x,y=zip(*d_items)
    fig=plt.figure()
    plt.plot(x,y,style)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titleStr)
    if not saveLoc=='':
        fig.savefig(os.path.join(saveLoc,titleStr)+'.png')
    return x,y


def plotDictWError(d,err,xlab='',ylab='',titleStr='',style='',saveLoc=''):
    x=[]
    y=[]
    errors=[]
    for key in d.keys():
        x.append(key)
        y.append(d[key])
        errors.append(err[key])
    x=np.array(x)
    y=np.array(y)
    errors=np.array(errors)
    fig=plt.figure()
    plt.errorbar(x,y,yerr=errors,fmt=style,capsize=5)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titleStr)
    if not saveLoc=='':
        fig.savefig(os.path.join(saveLoc,titleStr)+'.png')
    return x,y