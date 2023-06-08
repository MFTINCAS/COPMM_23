# -*- coding: utf-8 -*-
"""
Created on Wed May 04 22:50:48 2011

@author: FRANCES
"""
import numpy as np
import scipy.interpolate as spint
import matplotlib.pylab as plt
import scipy
from KDEpy import FFTKDE
from utils.ISJ_bandwidth import improved_sheather_jones

def distempcont_bs2(Xd,wd,opt_max_EMP):
    """
    Esempio di distribuzione empirica continua
    """
    
    Nd = len(Xd)
    if np.isfinite(opt_max_EMP['distr_emp_bs2_bw']):
        Ne=opt_max_EMP['distr_emp_bs2_bw']
    #if opt_max_EMP['distr_emp_bs2_bw']=='ISJ':
        #bw=improved_sheather_jones(Xd.reshape(-1,1))
    else:
       Ne  = 1 + int( np.log(Nd))
       Ne =  1 + int( (Nd**(1/3)))
      
    if Ne%2 == 0:
       Ne = Ne+1
       opt_max_EMP['distr_emp_bs2_bw'] = Ne

 
    ecdf_max=True
    warn_ne=False
    while ecdf_max :
        Nd = len(Xd)
        
        Xdm = min(Xd)
        XdM = max(Xd)
        # TODO option ISJ
        #------------------------------------
        # if opt_max_EMP['distr_emp_bs2_bw']=='ISJ':
        #     Ne=np.int(np.ceil((XdM-Xdm)/bw))
        #     ix = np.linspace(Xdm-bw,XdM+bw,Ne)
        #     ix = np.sort(ix)
        #     Ne = len(ix)
        # else:
        #     h=(XdM-Xdm)*1e-2
        #     #h=(XdM-Xdm)*1e-8
        #     ix = np.linspace(Xdm-h,XdM+h,Ne)
        #     ix = np.sort(ix)
        #     Ne = len(ix)
        #------------------------------------
        h=(XdM-Xdm)*1e-2#Mazzia
        
        ix = np.linspace(Xdm-h,XdM+h,Ne)
        ix = np.sort(ix)
        Ne = len(ix)
        pc = np.zeros( Ne)   
        for i in range(Ne):
            pc[i] = np.sum( wd[ Xd <= ix[i]] )
       
        if np.sum(wd)> 0:
           pc = pc/np.sum(wd)
        else:
           warn_ne=True 
       # ix =  np.concatenate(([ ix[0]-(ix[1]-ix[0]) ], ix,[  ix[Ne-1]+(ix[Ne-1]-ix[Ne-2]) ]),axis=0)
       # pc =  np.concatenate(([ 0 ], pc,[1]),axis=0)
       
        linear = opt_max_EMP['distemp_bs2_linear']
       
        if linear:        
           fyt= spint.interp1d(ix,pc,kind='linear')
           Nx = ix.shape[0]
           hx =  (ix[1:(Nx)]-ix[0:(Nx-1)])
           ix2 = ix[0:(Nx-1)]+hx/2
           #ix3 = ix[0:(Nx-1)]+2*hx/4
           #ix4 = ix[0:(Nx-1)]+3*hx/4
           #ix2=np.sort(np.concatenate( (ix,ix2,ix3,ix4),axis=0)) 
           ix2=np.sort(np.concatenate( (ix,ix2),axis=0)) 
           x = np.copy(ix2)
           y = fyt(x)
        else:
           x = np.copy(ix) 
           y = np.copy(pc)
        Nx = x.shape[0]
        
        hx =  (x[1:(Nx)]-x[0:(Nx-1)])
        fy = (y[2:(Nx)]-y[0:(Nx-2)])/(hx[1:]+hx[0:(Nx-2)])
        fy = np.concatenate(  ( [ (y[1]-y[0])/hx[0] ] ,fy,[(y[Nx-1]-y[Nx-2])/hx[Nx-2] ] ), axis=0 )   
        cs = (y[1:]+y[0:(Nx-1)])/2.0 + (hx*(fy[1:]-fy[0:(Nx-1)]))/4.0
        cs = np.concatenate(([y[0]],cs,[y[Nx-1]]),axis=0)
        xs = np.concatenate(([x[0]],[x[0]],x,[x[Nx-1]],[x[Nx-1]]),axis=0)
        fyt2=scipy.interpolate.BSpline(xs, cs, 2, extrapolate=True, axis=0)
        
        fyy = (y[2:Nx]-2*y[1:(Nx-1)]+y[0:(Nx-2)])/(hx[1:]**2);
        #fyy = np.concatenate(  ( [ (fy[1]-fy[0])/hx[0] ] ,fyy,[(fy[Nx-1]-fy[Nx-2])/hx[Nx-2] ]  ), axis=0 )   
        fyy = np.concatenate(  ( [ 0 ] ,fyy,[ 0]  ), axis=0 )   
    
        csyy = (fy[1:]+fy[0:(Nx-1)])/2.0 - (hx*(fyy[1:]-fyy[0:(Nx-1)]))/4.0
        csyy = np.concatenate(([ fy[0] ],csyy,[ fy[Nx-1] ]),axis=0)
        xsyy = np.concatenate(([x[0]],[x[0]],x,[x[Nx-1]],[x[Nx-1]]),axis=0)
        fyydt2=scipy.interpolate.BSpline(xs, csyy, 2, extrapolate=True, axis=0)
        fyyt22 = scipy.interpolate.splantider(fyydt2)
      
        
        # fd2 = scipy.interpolate.splder(fyt2,n=1)
        # y = fd2(x)
        # Nx = x.shape[0]
        
        # hx =  (x[1:(Nx)]-x[0:(Nx-1)])
        # fy = (y[2:(Nx)]-y[0:(Nx-2)])/(hx[1:]+hx[0:(Nx-2)])
        # fy = np.concatenate(  ( [ (y[1]-y[0])/hx[0] ] ,fy,[(y[Nx-1]-y[Nx-2])/hx[Nx-2] ] ), axis=0 )   
      
        # cs = (y[1:]+y[0:(Nx-1)])/2.0 + (hx*(fy[1:]-fy[0:(Nx-1)]))/4.0
        # cs = np.concatenate(([y[0]],cs,[y[Nx-1]]),axis=0)
        # xs = np.concatenate(([x[0]],[x[0]],x,[x[Nx-1]],[x[Nx-1]]),axis=0)
        # fd22=scipy.interpolate.BSpline(xs, cs, 2, extrapolate=True, axis=0)
        
        # fyt22 = scipy.interpolate.splantider(fd22)
        #epdf = fd22(Xd)
        #ecdf = fyt22(Xd)#/fyt22(np.max(Xd))
        epdf = fyydt2(Xd)
        if fyyt22(XdM) > 1.0001:
           ecdf = fyyt22(Xd)/fyyt22(XdM)
           ecdf_max=False      
           #Ne = Ne+1
           #opt_max_EMP['distr_emp_bs2_bw'] = Ne
           print(Ne,fyyt22(XdM))
        else:
           ecdf = fyyt22(Xd)
           ecdf_max=False
      
    
    
    # plot=True
    # if plot:
    #     t = np.linspace(ix[0],ix[Ne-1],500)
    #     plt.figure(1)
    #     plt.plot(ix,pc,'.',t,fyt22(t),'--' )
    #     plt.show()
    #     plt.hist(Xd,bins=50,density=True)
    #     plt.plot(t,fd22(t),'--',t,fd2(t),'.' )
    #     plt.plot(t,fyydt2(t),'--',t,fd2(t),'.' )
    #     plt.show()
    
    
    return (epdf,ecdf,warn_ne)

