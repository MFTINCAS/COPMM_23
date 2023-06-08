
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:17:25 2021
EXPECTATION STEP based on EMPIRICAL marginal calculation
is the sama of expectation step ECM if the marginal are normal
@author: Cristiano
"""

import numpy as np
from copulae.elliptical import NormalCopula
from scipy.stats import norm
from copulae.core import pseudo_obs as pobs
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import random


def expectation_step_EMP(self,X, clusters):
    
    n_features=X.shape[1]
    n_cluster=len(clusters)
    
    resp = np.zeros((X.shape[0], n_cluster))
    resp_sum=np.zeros(X.shape[0],)
    
    #gc_pdf = np.zeros((X.shape[0], 1), dtype=np.float64)
    #totals = np.zeros((X.shape[0], 1), dtype=np.float64)

    '''the next step is for empirical evaluation'''
    
    for cluster in clusters:
        resp_nk_marginal=np.zeros((X.shape[0],X.shape[1]))
        pi_k = cluster['pi_k']
        cop_k=cluster['cop_k_best']
        emp_k_pdf=np.copy(cluster['emp_pdf'])
        ecdf_k=np.copy(cluster['emp_cdf'])
        #ecdf_k=pobs(emp_k_pdf)

        '''
        
        if self.temp_iter==0:
            aa=np.arange(self.n_sample)
            temp_index=np.delete(aa,cluster['index_cluster_'], axis=0)
            l=len(temp_index)
            bb=np.repeat(1e-4,l).reshape(-1,1)
            bb=np.concatenate((bb,temp_index.reshape(-1,1)),axis=1)
            
            
              
    
            cop_pdf_k=cop_k.pdf(cop_k.random(self.n_sample))
            #cop_pdf_k=cop_k.pdf(ecdf_k)
            #cop_pdf_k=np.concatenate((cop_pdf_k.reshape(-1,1),cluster['index_cluster_'].reshape(-1,1)), axis=1)
            #mio=np.concatenate((cop_pdf_k,bb),axis=0)
            #mio=mio[np.argsort(mio[:,1])]
            
            #cluster['cop_pdf_k']=mio[:,0]
            cluster['cop_pdf_k']=cop_pdf_k.copy()
            
        else:
            cop_pdf_k=cop_k.pdf(cop_k.random(self.n_sample))
        '''   
        
        cop_pdf_k=cop_k.pdf(ecdf_k)
        
        cluster['cop_pdf_k']=cop_pdf_k.copy()
        #mvdgc='multi variate distribution gaussian copula'
        kdepdf_mult=np.prod(emp_k_pdf,axis=1) 
        mvdgc = np.array(cluster['cop_pdf_k']*kdepdf_mult).reshape(-1,1)

        resp_nk = np.array((pi_k * mvdgc).astype(np.float64)).reshape(-1,1)
        plt.scatter(X[:,0],X[:,1], c=mvdgc, s=5)
        #plt.colorbar()
        
        #aggiustare questa parte sulle approssimazionni
        #resp_nk=np.around(resp_nk,6)
        #resp_nk=np.where(resp_nk>=1, 1-1e-7,resp_nk)
        #resp_nk=np.where(resp_nk==0, 1e-6,resp_nk)
        resp_nk=np.where(resp_nk<=1e-8, 1e-8,resp_nk)
        
        cluster['resp_nk'] = resp_nk 
        
        cluster['resp_nk_marginal']=resp_nk_marginal
        
    # normalize across columns to make a valid probability       
    for k in range(n_cluster):
        resp[:,k] = clusters[k]['resp_nk'].transpose()
            
    resp_sum=np.sum(resp, axis=1).reshape(-1,1)
    resp/=resp_sum
    #aggiustare qui per le approssimazioni
    #resp=np.around(resp,7)
    #resp=np.where(resp==0, 1e-7,resp)
    #controllo su dataset su overlap clustering
    
    #2023 maggio: controllare se le tre righe sotto vanno prima del controllo o dopo
    for k in range(n_cluster):
        clusters[k]['resp_nk']=resp[:,k]
        clusters[k]['resp_sum']=resp_sum
    
    aa=resp
    #cc=np.argmax(aa,1)
    bb=np.where(np.abs(aa[:,0]-aa[:,1])<5e-1)[0]
    #bb=np.where(np.var(aa,1)<1e-3)[0]#for dim of resp>2????
    for i in bb:
        rand=random.choice(aa[i,:])
        ind=np.where(aa[i,:]==rand)[0]
        aa[i,ind]=rand*2
        #aa[i,ind-1]=rand*0.2
        
        
    resp=np.copy(aa)
    '''   
    eval final posterior probability p(z|X) 
    and restore for all cluster in clusters
    '''
    # for k in range(n_cluster):
    #     clusters[k]['resp_nk']=resp[:,k]
    #     clusters[k]['resp_sum']=resp_sum
    plt.scatter(X[:,0],X[:,1], c=np.argmax(resp,1), s=5)
    plt.show()
    return resp
