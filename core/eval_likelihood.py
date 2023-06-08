
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
import math

def eval_likelihood(X, clusters):
    n_features=X.shape[1]
    log_normpdf = np.zeros((X.shape))
    n_cluster=len(clusters)
    likelihood=np.zeros((X.shape[0],n_cluster))
    labels=[]
    k=0
    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mean_k']
        std_k = cluster['std_k']
        gc_k=cluster['cop_k_best']
        norm_k_cdf=cluster['marginal_cdf']
        resp_nk=cluster['resp_nk']
        '''
        calc of gamma(znk) within each cluster that is responsability
        resp is an array with length of (Nxd) i.e. X.shape[0]xd
        '''
        for d in range(n_features):
            l=norm.pdf(X[:,d], mu_k[d], std_k[d])
            #l=np.around(l,8)
            l=np.where(l<=1e-8, 1e-8,l)
            #l=np.where(l<=0, 1e-08,l)
            log_normpdf[:,d]=np.log(l)
            
        #likelihood=np.log(np.array(cluster['resp_sum']))
        #likelihood=np.where(math.isnan(likelihood),0,likelihood)
        #likelihood[:,k] = -(np.log(pi_k)+gc_k.pdf(norm_k_cdf, log=True)+np.sum(log_normpdf, axis=1))
        labels.append(cluster['resp_nk'])
        k=k+1
    likelihood=np.log(np.array(cluster['resp_sum']))
    labels=np.argmax(np.stack(labels).transpose(),axis=1)
    likelihood=np.sum(likelihood,axis=0).reshape(-1,1)
    metric=davies_bouldin_score(X, labels)
    metric2=metrics.calinski_harabasz_score(X, labels)
    
    return np.sum(likelihood), likelihood, metric, metric2


'''
import numpy as np


def get_likelihood(X, clusters):
    for cluster in clusters:
        sample_likelihoods = np.log(np.array(cluster['totals']))
        
    return np.sum(sample_likelihoods), sample_likelihoods'''