"""
Created on Mon Feb  8 12:17:25 2021
EXPECTATION STEP IFM 
is the sama of expectation step ECM if the marginal are normal
@author: Cristiano
"""

import numpy as np
from copulae.elliptical import NormalCopula
from scipy.stats import norm

def expectation_step_IFM(self,X, clusters):
    n_features=self.n_features
    n_cluster=self.n_clusters
    normpdf = np.zeros((X.shape))
    resp = np.zeros((X.shape[0], n_cluster))
    resp_sum=np.zeros(X.shape[0],)
    #gc_pdf = np.zeros((X.shape[0], 1), dtype=np.float64)
    #totals = np.zeros((X.shape[0], 1), dtype=np.float64)
    
    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mean_k']
        std_k = cluster['std_k']
        cop_k=cluster['cop_k_best']
        norm_k_cdf=cluster['marginal_cdf']
        #norm_k_pdf=cluster['marginal_pdf']
        
        '''
        calc of gamma(znk) within each cluster that is responsability
        resp is an array with length of (Nxd) i.e. X.shape[0]xd
        '''
        for d in range(n_features):
            normpdf[:,d]=norm.pdf(X[:,d], mu_k[d], std_k[d])
        '''
        multiply the column by row
        for have the product of marginal f1(x1)*f2(x2)*...*fd(x_d)
        '''
        normpdf_mult=np.prod(normpdf,axis=1) 
        '''
        evaluate density of gaussian copula
        '''
        cop_pdf_k=cop_k.pdf(norm_k_cdf)
        cluster['cop_k_pdf']=cop_pdf_k
        #mvdgc='multi variate distribution gaussian copula'
        mv_pdf_cop = np.array([cop_pdf_k*normpdf_mult])
        
        resp_nk = np.array((pi_k * mv_pdf_cop).astype(np.float64)).reshape(-1,1)
        #resp_nk=np.around(resp_nk,6)
        #resp_nk=np.where(resp_nk>=1, 1-1e-6,resp_nk)
        resp_nk=np.where(resp_nk==0, 1e-6,resp_nk)
        cluster['resp_nk'] = resp_nk 
    
    # normalize across columns to make a valid probability       
    for k in range(n_cluster):
        resp[:,k] = clusters[k]['resp_nk'].transpose()
        
    resp_sum=np.sum(resp, axis=1).reshape(-1,1)
    cluster['resp_sum']=resp_sum 
    resp/=resp_sum
    #resp=np.around(resp,7)
    #resp=np.where(resp==0, 1e-7,resp)
    '''   
    eval final posterior probability p(z|X) 
    and restore for all cluster in clusters
    '''
    for k in range(n_cluster):
        clusters[k]['resp_nk']=resp[:,k]   
        clusters[k]['resp_sum']=resp_sum 
    return resp