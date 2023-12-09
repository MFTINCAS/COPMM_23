# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:07:25 2023

@author: Cristiano
"""

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
from utils.distr_emp_bs_23_new import BSemp
#from utils.old_emp_distribution.distr_emp_bs import distempcont_bs2


'''
Questo codice usa il nuovo file per l'approssimazione delle densità 'distr_emp_bs_23_new'
'''

def emp_distr_new(X,resp_nk,index_cluster,k):
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
        
        
        # resp_nk=resp_nk.reshape(-1,1)
        for d in range(n_features): 
              wd=np.copy(resp_nk[:X.shape[0]])
              sk = wd[ np.where(index_cluster==k)[0]].shape[0]
              nk = X.shape[0]
              aldm= 1e-7#Mazzia
              #aldm=1e-2
              wd[ np.where(index_cluster!=k)[0]]=wd[ np.where(index_cluster!=k)[0]]*aldm
              nk1=len(X[:,d])
       
              [_,kde_pdf[:,d], ecdf[:,d],warn_ne]= BSemp(X[:,d],wd,bw=1+int(nk1**(1/3)),sxeval=X[:,d])
              if  warn_ne:            
                 wd[ np.where(index_cluster!=k)[0]]=wd[ np.where(index_cluster!=k)[0]]+1         
                 #[_,_,kde_pdf[:,d], ecdf[:,d],warn_ne]= distemp_new(X[:,d],wd,bw=1+int(nk1**(1/3)),sxeval=X[:,d]) 
                 [_,kde_pdf[:,d], ecdf[:,d],warn_ne]= BSemp(X[:,d],wd,bw=1+int(nk1**(1/3)),sxeval=X[:,d]) 
                 #[kde_pdf[:,d], ecdf[:,d],warn_ne]= distempcont_bs2(X[:,d],wd)  
              
        ecdf=np.where(ecdf>=1-1e-5, 1-1e-5,ecdf)#1-1e-8
        ecdf=np.where(ecdf<=1e-5, 1e-5,ecdf)#1e-8
        
        #kde_pdf=np.where(kde_pdf<=1-1e-8, 1-1e-8,kde_pdf)#aggiunto da me
        
        return (kde_pdf,ecdf)
    
def emp_distr_kdepy(X,resp_nk,index_cluster,k):
        n_features=X.shape[1]
        kde_pdf=np.zeros((X.shape[0] ,n_features))
        kde_cdf=[]
        kdes_pdf=np.zeros((X.shape[0] ,n_features))
        ecdf=np.zeros((X.shape[0],n_features))
        kde_pdf1=np.zeros((X.shape[0] ,n_features))
        ecdf1=np.zeros((X.shape[0],n_features))
      
        kde=[]
        kdes=[]
        ecdf_obj=[]
        ecdf_o=[]
        
        
        # resp_nk=resp_nk.reshape(-1,1)
        for d in range(n_features): 
              wd=np.copy(resp_nk[:X.shape[0]])
              sk = wd[ np.where(index_cluster==k)[0]].shape[0]
              nk = X.shape[0]
              aldm= 1e-1#Mazzia
              #aldm=1e-2
              wd[ np.where(index_cluster!=k)[0]]=wd[ np.where(index_cluster!=k)[0]]*aldm
              nk1=len(X[:,d])
              kde = FFTKDE(kernel='gaussian', bw=1+int(nk1**(1/3))).fit(X[:,d],weights=wd)
              Xdm = min(X[:,d])
              XdM = max(X[:,d])
              xkde, ykde=kde.evaluate(X[:,d].size)

             
              kdex=make_interp_spline(xkde,ykde,k=2)
              kde_pdf[:,d]=splev(X[:,d],kdex)
              kde_cdf.append(scipy.interpolate.splantider(kdex))
              ecdf[:,d] = kde_cdf[d](X[:,d])
      
              
        ecdf=np.where(ecdf>=1-1e-5, 1-1e-5,ecdf)#1-1e-8
        ecdf=np.where(ecdf<=1e-5, 1e-5,ecdf)#1e-8
        
        #kde_pdf=np.where(kde_pdf<=1-1e-8, 1-1e-8,kde_pdf)#aggiunto da me
        
        return (kde_pdf,ecdf)
