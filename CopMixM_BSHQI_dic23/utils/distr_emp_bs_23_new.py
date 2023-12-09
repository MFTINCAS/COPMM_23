# -*- coding: utf-8 -*-

import numpy as np
import scipy.interpolate as spint
import matplotlib.pylab as plt
import scipy
#from utils.ISJ_bandwidth import improved_sheather_jones
from sklearn.model_selection import cross_val_score
import pandas as pd
import time

def empirical_cdf(s: pd.Series, n_bins: int = 100):
    # Sort the data into `n_bins` evenly spaced bins:
    discretized = pd.cut(s, n_bins, precision=16)
    # Count the number of datapoints in each bin:
    bin_counts = discretized.value_counts().sort_index().reset_index()
    # Calculate the locations of each bin as just the mean of the bin start and end:
    bin_counts["loc"] =  pd.IntervalIndex(bin_counts["index"]).right
    # Compute the CDF with cumsum:
    return bin_counts.set_index("loc").iloc[:, -1].cumsum()


def BSemp(Xd,wd=None, bw=100, sxeval=None, leftbw=0, rigthbw=0):
    """
    Esempio di distribuzione empirica continua
    """
    
    Nd = len(Xd)
    Ne = bw

    if sxeval is None:
       sxeval=Xd
     
      
    if Ne%2 == 0:
       Ne = Ne+1
    
    Nd = len(Xd)
    Xdm = min(Xd)
    XdM = max(Xd)
    
     #------------------------------------
    h=(XdM-Xdm)*1e-2#  
    #Xd = np.sort(Xd)
    ix = np.linspace(Xdm-h,XdM+h,Ne)
    #ix=np.linspace(5-2,5+2,Ne)
    hb = ix[2]-ix[1]
    
    warn_ne=False
    
       
    if wd is None or np.sum(np.diff(wd))==0:# ricorda di togliere hash
       pcc = empirical_cdf(Xd,Ne-1)   
       #ix = np.array(pcc.index)
       #ix = np.concatenate(([Xdm],ix),axis=0)
       pc = np.array(pcc)
       if leftbw==0  and rigthbw==0:
          nadd = 4
          ix = np.linspace(Xdm-2*hb,XdM+2*hb,Ne+4)
          pc = np.concatenate(([0,0,0],pc/Nd,[1,1]),axis=0)
       elif    leftbw==0  and rigthbw==1:
          nadd = 2
          ix = np.linspace(Xdm-2*hb,XdM+h,Ne+2)
          pc = np.concatenate(([0,0,0],pc/Nd),axis=0)
       elif    leftbw==1  and rigthbw==0:
          nadd = 2 
          ix = np.linspace(Xdm-h,XdM+2*hb,Ne+2)
          pc = np.concatenate(([0],pc/Nd,[1,1]),axis=0)
       elif    leftbw==1  and rigthbw==1:
          nadd = 0
          ix = np.linspace(Xdm-h,XdM+h,Ne)
          pc = np.concatenate(([0],pc/Nd),axis=0)
     
      # pc = np.concatenate(([0],pc),axis=0)/Nd
       print(pc.size,Nd)
       
       
    
       
    else:
      
       
       if leftbw==0  and rigthbw==0:
          nadd = 4
          ix = np.linspace(Xdm-2*hb,XdM+2*hb,Ne+4)
       elif    leftbw==0  and rigthbw==1:
          nadd = 2
          ix = np.linspace(Xdm-2*hb,XdM+h,Ne+2)
       elif    leftbw==1  and rigthbw==0:
          nadd = 2 
          ix = np.linspace(Xdm-h,XdM+2*hb,Ne+2)
       elif    leftbw==1  and rigthbw==1:
          nadd = 0
          ix = np.linspace(Xdm-h,XdM+h,Ne)
          
       #ix=np.linspace(5-2,5+2,Ne +nadd)  
       #wd=np.ones(Ne+nadd)
       pc = np.zeros(Ne+nadd)     
       for i in range(Ne+nadd):
          pc[i] = np.sum( wd[ Xd <= ix[i]] )#cumulata discreta
          
      
       if np.sum(wd)> 0:
          pc = pc/np.sum(wd)
       else:
          warn_ne=True 
    
      
    x = np.copy(ix) 
    y = np.copy(pc)           
    Nx = x.shape[0]
    hx =  (x[1:(Nx)]-x[0:(Nx-1)]) 
    xs = np.concatenate(([x[0]],[x[0]],x,[x[Nx-1]],[x[Nx-1]]),axis=0)
    
    csyy = (y[1:]-y[0:(Nx-1)])/hx
    csyy[0]=((y[1]-y[0]))/hx[0]
    csyy[Nx-2]=((y[Nx-1]-y[Nx-2]))/hx[Nx-2]   
    csyy = np.concatenate(([ csyy[0] ],csyy,[ csyy[Nx-2] ]),axis=0)
     
    bsh_epdf = scipy.interpolate.BSpline(xs, csyy, 2, extrapolate=True)
    bsh_ecdf = scipy.interpolate.splantider(bsh_epdf)
     
    nd = np.size(sxeval)
    epdf = np.zeros(nd)
    if nd>1:
       sxeval0 = sxeval.copy()
    else:
       sxeval0 = sxeval
   
    if leftbw==1:
      sxeval0 = sxeval0[np.where( sxeval >=Xdm)]
      indin = np.min(np.where( sxeval >=Xdm))
    else:
      indin=0
    if rigthbw==1:
       sxeval0 = sxeval0[np.where( sxeval <=XdM)]
    epdf0 = bsh_epdf(sxeval0)
    nd0 = np.size(epdf0)
    epdf[indin:(nd0+indin)] = epdf0
    

    if bsh_ecdf(XdM) > 1.0001:
      drv  = bsh_ecdf(XdM)
      ecdf0 = (bsh_ecdf(sxeval0))/drv
    else:
      ecdf0 = bsh_ecdf(sxeval0)
      nd0 = np.size(ecdf0)
    ecdf = np.zeros(nd)
    ecdf[indin:(nd0+indin)] = ecdf0
    ecdf[(nd0+indin):nd] = 1
    
    bsh_data={"Xdm":Xdm,"XdM":XdM,"leftbw":leftbw,"rigthbw":rigthbw,"epdf":bsh_epdf,"ecdf":bsh_ecdf}
    
    
    return (bsh_data,epdf,ecdf,warn_ne)

def BSemp_pdf_eval(bsh_data,x):
        bsh_epdf = bsh_data["epdf"]
        Xdm = bsh_data["Xdm"]
        XdM = bsh_data["XdM"]
        leftbw = bsh_data["leftbw"]
        rigthbw = bsh_data["rigthbw"]
        sxeval = x
        nd = np.size(sxeval)
        epdf = np.zeros(nd)
        if nd>1:
           sxeval0 = sxeval.copy()
        else:
           sxeval0 = sxeval
       
        if leftbw==1:
          sxeval0 = sxeval0[np.where( sxeval >=Xdm)]
          indin = np.min(np.where( sxeval >=Xdm))
        else:
          indin=0
        if rigthbw==1:
           sxeval0 = sxeval0[np.where( sxeval <=XdM)]
        epdf0 = bsh_epdf(sxeval0)
        nd0 = np.size(epdf0)
        epdf[indin:(nd0+indin)] = epdf0
        return (epdf)
    
    
def BSemp_cdf_eval(bsh_data,x):
        bsh_ecdf = bsh_data["ecdf"]
        Xdm = bsh_data["Xdm"]
        XdM = bsh_data["XdM"]
        leftbw = bsh_data["leftbw"]
        rigthbw = bsh_data["rigthbw"]
        sxeval = x
        nd = np.size(sxeval)
        if nd>1:
           sxeval0 = sxeval.copy()
        else:
           sxeval0 = sxeval  
       
        if leftbw==1:
          sxeval0 = sxeval0[np.where( sxeval >=Xdm)]
          indin = np.min(np.where( sxeval >=Xdm))
        else:
          indin=0
        if rigthbw==1:
           sxeval0 = sxeval0[np.where( sxeval <=XdM)]

        
        if bsh_ecdf(XdM) > 1.0001:
          drv  = bsh_ecdf(XdM)
          ecdf0 = (bsh_ecdf(sxeval0))/drv
        else:
          ecdf0 = bsh_ecdf(sxeval0)
          nd0 = np.size(ecdf0)
        ecdf = np.zeros(nd)
        ecdf[indin:(nd0+indin)] = ecdf0
        ecdf[(nd0+indin):nd] = 1
        return (ecdf)
         
        

    