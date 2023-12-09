# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:40:25 2021

@author: Cristiano
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:00:20 2020

@author: Cristiano
"""

import numpy as np
from scipy.stats import norm
from copulae.stats import multivariate_normal as mvn, norm
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score

def eval_likelihood_EMP(self,X, clusters):
    n_features=X.shape[1]
    log_kdepdf = np.zeros((X.shape))
    n_cluster=len(clusters)
    likelihood=np.zeros((X.shape[0],n_cluster))
    k=0
    labels=[]
    for cluster in clusters:
        pi_k = cluster['pi_k']
        #if self.cop_fam=='all':
        cop_k=cluster['cop_k_best']
        #else:
            #cop_k=cluster['cop_k']
        emp_k_pdf=np.copy(cluster['emp_pdf'])
        resp_nk=np.copy(cluster['resp_nk'])
        ecdf=np.copy(cluster['emp_cdf'])
        
        for d in range(n_features):
           # l=norm.pdf(X[:,d], al_k[d], bet_k[d])
            l=emp_k_pdf[:,d]
            #l=np.around(l,8)
            l=np.where(l<=1e-8, 1e-8,l)
            log_kdepdf[:,d]=np.log(l)
       # kde_k_pdf=np.where(kde_k_pdf==0, 1e-08,kde_k_pdf)
        #log_kdepdf=np.log(kde_k_pdf)
        #likelihood=np.log(np.array(cluster['resp_sum']))
        #likelihood[:,k] = resp_nk*(np.log(pi_k)+cop_k.pdf(ecdf,log=True)+np.sum(log_kdepdf, axis=1))
        labels.append(cluster['resp_nk'])
        k=k+1
    likelihood=np.log(np.array(cluster['resp_sum']))
    #likelihood=np.sum(likelihood,axis=1).reshape(-1,1)
    labels=np.argmax(np.stack(labels).transpose(),axis=1)
    likelihood=np.sum(likelihood,axis=0).reshape(-1,1)
    metric=davies_bouldin_score(X, labels.ravel())
    metric2=metrics.calinski_harabasz_score(X, labels.ravel())
    #----------------------------------------------------controlla
    #likelihood[likelihood>-1e-2]=-1e-8
    
    return np.sum(likelihood), likelihood, metric, metric2


'''
import numpy as np


def get_likelihood(X, clusters):
    for cluster in clusters:
        sample_likelihoods = np.log(np.array(cluster['totals']))
        
    return np.sum(sample_likelihoods), sample_likelihoods'''