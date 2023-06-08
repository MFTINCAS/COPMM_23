# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:59:57 2021

@author: Cristiano
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:56:33 2021

@author: Cristiano
"""

import numpy as np
import math
import scipy.interpolate as spint
import matplotlib.pylab as plt
import scipy
from KDEpy import FFTKDE
from scipy.interpolate import splrep, splder, splantider, splev, make_interp_spline
from statsmodels.nonparametric.bandwidths import bw_silverman
from utils.distr_emp_bs import distempcont_bs2 


def emp_distr(X,al_k,bet_k,resp_nk,index_cluster,k, opt_max_EMP):
        n_features=X.shape[1]
        kde_pdf=np.zeros((X.shape[0] ,n_features))
        kdes_pdf=np.zeros((X.shape[0] ,n_features))
        ecdf=np.zeros((X.shape[0],n_features))
        kde_pdf1=np.zeros((X.shape[0] ,n_features))
        ecdf1=np.zeros((X.shape[0],n_features))
      
        kde=[]
        kdes=[]
        ecdf_obj=[]
        ecdf_o=[]
        
        linear= opt_max_EMP['emp_distr_linear']
        allbs = opt_max_EMP['emp_distr_allbs']
        # resp_nk=resp_nk.reshape(-1,1)
        for d in range(n_features): 
           wd=np.copy(resp_nk[:X.shape[0]])
           sk = wd[ np.where(index_cluster==k)[0]].shape[0]
           nk = X.shape[0]
           aldm= 1e-3#Mazzia
           #aldm=1e-2
           wd[ np.where(index_cluster!=k)[0]]=wd[ np.where(index_cluster!=k)[0]]*aldm
              
     
           
           
           if allbs:
               
             [kde_pdf[:,d], ecdf[:,d],warn_ne]= distempcont_bs2(X[:,d],wd, opt_max_EMP)  
             if  warn_ne:            
                 wd[ np.where(index_cluster!=k)[0]]=wd[ np.where(index_cluster!=k)[0]]+1         
                 [kde_pdf[:,d], ecdf[:,d],warn_ne]= distempcont_bs2(X[:,d],wd, opt_max_EMP)  
          
                 
         
           else:
             kde_=FFTKDE(kernel='gaussian', bw="ISJ").fit(X[:,d],weights= wd)
             Nx = 2**11
             x,y=kde_.evaluate(Nx)
           
             if linear:
                kde.append(make_interp_spline(x,y,k=1))
             else:                     
             #kde.append(make_interp_spline(x,y,bc_type="clamped",k=3))
             # spline quasi-interpolante di grado due di Hermite
             # derivate prime approssimate con le differenze centrali  nei nodi interni
             # e con le differenze in avanti e all'indietro ai bordi
             # il quasi interpolante non funziona con il kernel box
                hx =  (x[1:(Nx)]-x[0:(Nx-1)])
                fy = (y[2:(Nx)]-y[0:(Nx-2)])/(hx[1:]+hx[0:(Nx-2)])
                fy = np.concatenate(  ( [(y[1]-y[0])/hx[0]] ,fy,[ (y[Nx-1]-y[Nx-2])/hx[Nx-2]] ), axis=0 )  
             
                cs = (y[1:]+y[0:(Nx-1)])/2.0 - (hx*(fy[1:]-fy[0:(Nx-1)]))/4.0
                cs = np.concatenate(([y[0]],cs,[y[Nx-1]]),axis=0)
                xs = np.concatenate(([x[0],x[0]],x,[x[Nx-1],x[Nx-1]]),axis=0)
                
            
                
                kde.append(scipy.interpolate.BSpline(xs, cs, 2, extrapolate=True, axis=0))
                
            
           
             kde_pdf[:,d]=splev(X[:,d],kde[d] )
           #  [kde_pdf[:,d], ecdf[:,d]]= de.distempcont_bs2(X[:,d],wd)  
            # plt.plot(x,splev(x,kde[d]) )
            # plt.show()
            
        if not allbs:
          ecdf_obj = [ splantider(kde[d]) for d in range(n_features) ]   
           
          for d in range(n_features):
            XdM = np.max(X[:,d])
            ecdfXdM=splev(XdM,ecdf_obj[d])
            if ecdfXdM > 1:
               ecdf[:,d]=splev(X[:,d],ecdf_obj[d])/ecdfXdM
            else:
               ecdf[:,d]=splev(X[:,d],ecdf_obj[d])
  
              # plt.plot(x,splev(x,ecdf_obj[d]) )
            # plt.show()
           
      
            
       # ecdf=np.around(ecdf,10) #questa introduce rumore nella loglikelihood
        #nn=X.shape[0]
        #pig=math.pi
        #delta=1/((4*nn)**(1/4)*(pig*(np.log(nn))**(1/2)))
        #ecdf=np.where(ecdf>1-delta, 1-delta,ecdf)
        #ecdf=np.where(ecdf<delta, delta,ecdf)
        
        
        ecdf=np.where(ecdf>=1-1e-6, 1-1e-6,ecdf)#1-1e-8
        ecdf=np.where(ecdf<=1e-6, 1e-6,ecdf)#1e-8
        
        #kde_pdf=np.where(kde_pdf<=1-1e-1, 1-1e-1,kde_pdf)#aggiunto da me
        
        return (kde_pdf,ecdf)

# def distempcont_bs2(Xd,wd,opt_max_EMP):
#     """
#     Esempio di distribuzione empirica continua
#     """
    
#     Ne=opt_max_EMP['distr_emp_bs2_bw']#bw 
#     Nd = len(Xd)
    
    
    
#     Xdm = min(Xd)
#     XdM = max(Xd)
#     h=min( (XdM-Xdm)*1e-4,1e-2)
#     ix = np.linspace(Xdm-h,XdM+h,Ne)
#     ix = np.sort(ix)
#     Ne = len(ix)
     
#     pc = np.zeros( Ne)   
#     for i in range(Ne):
#         pc[i] = np.sum( wd[ Xd <= ix[i]] )
   
#     pc = pc/np.sum(wd)
  
#     linear = opt_max_EMP['distemp_bs2_linear']
   
#     if linear:        
#        fyt= spint.interp1d(ix,pc,kind='linear')
#        Nx = ix.shape[0]
#        hx =  (ix[1:(Nx)]-ix[0:(Nx-1)])
#        ix2 = ix[0:(Nx-1)]+hx/4
#        ix3 = ix[0:(Nx-1)]+2*hx/4
#        ix4 = ix[0:(Nx-1)]+3*hx/4
#        ix2=np.sort(np.concatenate( (ix,ix2,ix3,ix4),axis=0)) 
#        x = np.copy(ix2)
#        y = fyt(x)
#     else:
#        x = np.copy(ix) 
#        y = np.copy(pc)
#     Nx = x.shape[0]
    
#     hx =  (x[1:(Nx)]-x[0:(Nx-1)])
#     fy = (y[2:(Nx)]-y[0:(Nx-2)])/(hx[1:]+hx[0:(Nx-2)])
#     fy = np.concatenate(  ( [ (y[1]-y[0])/hx[0] ] ,fy,[(y[Nx-1]-y[Nx-2])/hx[Nx-2] ] ), axis=0 )   
#     cs = (y[1:]+y[0:(Nx-1)])/2.0 + (hx*(fy[1:]-fy[0:(Nx-1)]))/4.0
#     cs = np.concatenate(([y[0]],cs,[y[Nx-1]]),axis=0)
#     xs = np.concatenate(([x[0]],[x[0]],x,[x[Nx-1]],[x[Nx-1]]),axis=0)
#     fyt2=scipy.interpolate.BSpline(xs, cs, 2, extrapolate=True, axis=0)
    
    
#     fd2 = scipy.interpolate.splder(fyt2,n=1)
#     y = fd2(x)
#     #Nx = x.shape[0]
    
#     hx =  (x[1:(Nx)]-x[0:(Nx-1)])
#     fy = (y[2:(Nx)]-y[0:(Nx-2)])/(hx[1:]+hx[0:(Nx-2)])
#     fy = np.concatenate(  ( [ (y[1]-y[0])/hx[0] ] ,fy,[(y[Nx-1]-y[Nx-2])/hx[Nx-2] ] ), axis=0 )   
  
#     cs = (y[1:]+y[0:(Nx-1)])/2.0 + (hx*(fy[1:]-fy[0:(Nx-1)]))/4.0
#     cs = np.concatenate(([y[0]],cs,[y[Nx-1]]),axis=0)
#     xs = np.concatenate(([x[0]],[x[0]],x,[x[Nx-1]],[x[Nx-1]]),axis=0)
#     fd22=scipy.interpolate.BSpline(xs, cs, 2, extrapolate=True, axis=0)
    
#     fyt22 = scipy.interpolate.splantider(fd22)
#     epdf = fd22(Xd)
#     ecdf = fyt22(Xd)/fyt22(np.max(Xd))#da controllare!!
    
#     plot=False
#     if plot:
#         t = np.linspace(ix[0],ix[Ne-1],500)
#         plt.figure(1)
#         plt.plot(ix,pc,'.',t,fyt22(t),'--' )
#         plt.show()
#         plt.hist(Xd,bins=50,density=True)
#         plt.plot(t,fd22(t),'--',t,fd2(t),'.' )
#         plt.show()
    
    
#     ecdf=np.around(ecdf,8)
#     ecdf=np.where(ecdf>=0.99, 1-1e-02,ecdf)
#     ecdf=np.where(ecdf<=0.01, 1e-02,ecdf)
    
#     return (epdf,ecdf)