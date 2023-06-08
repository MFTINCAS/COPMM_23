"""
@author: Cristiano
"""

import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from scipy.stats import invgauss
from copulae.elliptical import GaussianCopula, StudentCopula
from copulae.archimedean import FrankCopula, ClaytonCopula, GumbelCopula
from copulae.copula.estimator import fit_copula
from utils.cop_family import cop_family
from utils.cop_family import cop_family_new
from scipy.stats import norm
from scipy.stats import truncnorm
from statsmodels.distributions.empirical_distribution import ECDF
from copulae.core import pseudo_obs as pobs
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from scipy.interpolate import interp1d, splrep, splder, splantider, splev, make_interp_spline
import random
from utils.mytools import split_random
from statsmodels.distributions.empirical_distribution import ECDF
from copulae.core import pseudo_obs
from core.eval_likelihood_EMP import eval_likelihood_EMP
from core.expectation_step_ECM import expectation_step_ECM
from core.expectation_step_IFM import expectation_step_IFM
from core.expectation_step_EMP import expectation_step_EMP
from core.maximization_step_ECM import maximization_step_ECM
from core.maximization_step_IFM import maximization_step_IFM
from core.maximization_step_EMP_KDEpy import maximization_step_EMP_KDEpy
from core.maximization_step_EMP_spline import maximization_step_EMP_spline
from sklearn.metrics.cluster import adjusted_mutual_info_score
import warnings
import hdbscan
from hdbscan import flat
from sklearn.mixture import GaussianMixture


#from Utils.initialize_cluster_random import initialize_clusters_random




def initialize_clusters_base(self, X):
    n_features=self.n_features
    len_dataset=self.n_sample
    clusters = []
    ecdf_obj=[]
    plot=False

            
    if self.init_clust=='k_means':
        #idx = np.arange(X.shape[0])
        # KMeans for initialization
        kmeans = KMeans(self.n_clusters,algorithm='elkan').fit(X)
    
        #store first mu_k std_k, pi_k in dict clusters 
        for k in range(self.n_clusters):  
            clusters.append({
                'marginals_': X[np.where(kmeans.labels_==k)],
                #'index_cluster_':np.where(kmeans.labels_==k)[0],
                'mean_k':np.mean(X[np.where(kmeans.labels_==k)],axis=0), 
                'std_k':np.std(X[np.where(kmeans.labels_==k)],axis=0),  
                'pi_k': 1.0 / self.n_clusters,
                'al_k': np.mean(X[np.where(kmeans.labels_==k)],axis=0)*0+0.99,
                'bet_k': np.std(X[np.where(kmeans.labels_==k)],axis=0)*0+1e-10,
            })
        self.first_label=kmeans.labels_
   

    elif self.init_clust=='random':
        rdmclust= split_random(len_dataset, self.n_clusters)
        labels=np.empty(len_dataset)
        for k in range(self.n_clusters):
            labels[rdmclust[k]]=k
        for k in range(self.n_clusters):
            clusters.append({
            'marginals_': X[np.where(labels==k)],
            'mean_k': np.mean(X[np.where(labels==k)],axis=0), 
            'std_k':np.std(X[np.where(labels==k)],axis=0),  
            'pi_k': 1.0 / self.n_clusters,
            'al_k': np.mean(X[np.where(labels==k)],axis=0)*0+0.99,
            'bet_k': np.std(X[np.where(labels==k)],axis=0)*0+1e-10
           # 'ce_k': np.std(X[np.where(kmeans.labels_==k)],axis=0)*0+0.99
        })
            
    #eval norm_cdf for all all features with mean and sd calculated before with K_Mean
    

    if type(self.cop_type_opt)==list:
        print('Fit copula step 0')
        print('first marginal inizialization with gaussian for pdf and for ecdf')
        for cluster in clusters:
            norm_k_cdf= np.zeros(X.shape)
            #fit marginal for the copula
            mu_k = cluster['mean_k']
            std_k = cluster['std_k']
            normpdf= np.zeros(X.shape)
            #memorizzo l'empirica cumulata sulla base della inizializazione fatta con una normale
            #memorizzo la kernel density sulla base dell'inizializazione fatta con la normale
            #se si vuole la cdf calcolata con la normaledecommentare la parte gialla sotto
            
            
            for d in range (n_features):
                norm_k_cdf[:,d]=norm.cdf(X[:,d],cluster['mean_k'][d],cluster['std_k'][d])
                
            #this is to have not error in evaluation of cop_log_like
            
            cluster['marginal_cdf']=norm_k_cdf
            #this is to have not error in evaluation of cop_log_like
            cluster['marginal_cdf']=np.where(cluster['marginal_cdf']>=0.999, 1-1e-8,cluster['marginal_cdf'])
            cluster['marginal_cdf']=np.where(cluster['marginal_cdf']<=0.001, 1e-8,cluster['marginal_cdf'])
            # norm_k_cdf=np.round(norm_k_cdf,7)
            # norm_k_cdf=np.where(norm_k_cdf>=0.99, 1-1e-2,norm_k_cdf)
            # norm_k_cdf=np.where(norm_k_cdf<=0.1, 1e-2,norm_k_cdf)
            # cluster['marginal_cdf']=norm_k_cdf.copy()
            if self.maximization_type=='EMP_spline':
                #memorizzo l'empirica cumulata sulla base della inizializazione fatta con una normale
                cluster['emp_cdf']=cluster['marginal_cdf']
                #memorizzo la kernel density sulla base dell'inizializazione fatta con la normale
                for j in range(n_features):
                    normpdf[:,j]=norm.pdf(X[:,j], mu_k[j], std_k[j])
                cluster['emp_pdf']=np.copy(normpdf)

            
            #se si vuole la pdf calcolata con fftkde decommentare la parte gialla sotto
            '''
            emp_pdf= np.zeros(X.shape)
            kde=[]
            for d in range(self.n_features):
                
                kde_=FFTKDE(kernel='gaussian', bw="silverman").fit(cluster['marginals_'][:,d])#,weights= cluster['ecdf'][:,d])
                x,y=kde_.evaluate(2**10)
                kde.append(interp1d(x, y, fill_value=1.0e-6,bounds_error=False))
                emp_pdf[:,d]=kde[d](X[:,d])
            cluster['emp_pdf']=np.copy(emp_pdf)
            '''
    
            
            # fit copula
            
            cop_fam=cop_family_new(self.cop_type_opt)
            aic=[]
            bic=[]
            cluster['cop_k_list']=[]
            k=0
            for name, cop in cop_fam:
                         
                cop=cop(dim=self.n_features)
                fit=fit_copula(cop, cluster['marginal_cdf'], x0=None, method='ml', optim_options={'method':'L-BFGS-B','options':{'ftol':1e-05, 'eps':1e-05,'maxfun':100, 'maxiter': 500,'gtol': 1.0e-3}}, verbose=1, scale=0.01)
                #fit=fit_copula(cop, pobs(cluster['marginals_']), x0=None, method='ml', optim_options={'method':'L-BFGS-B'}, verbose=2, scale=0.01)
                
                #CONTROLLA LA CONVERGENZA!!!! Metti un Controllo
                #if math.isnan(model.params)==False:
                cluster['cop_k_list'].append(cop)
                #log_like=(cop.log_lik((cluster['marginals_']), to_pobs=True))
                #log_like=-(cop.log_lik((cluster['marginal_cdf']), to_pobs=False))
                log_like=fit.log_lik #già negativa nel pacchetto copulae
     
                aic_value=-2*(log_like)+2*len(np.unique(cop.params))*len(cluster['marginals_'])
                bic_value=-2*(log_like)+len(np.unique(cop.params))*len(cluster['marginals_'])
                if math.isnan(aic_value)==False and math.isinf(aic_value)==False:
                    aic.append({'model':cop,
                                'aic':aic_value,
                                })
                    bic.append({'model':cop,
                                'bic':bic_value,
                                })
                elif len(cop_fam)==1:
                    raise ValueError('Set valid value for cop_type_opt')
            best_aic = min(aic, key=lambda x: x['aic'])
  
            best_aic = min(aic, key=lambda x: x['aic'])
            cluster['cop_k_best']=best_aic['model'] 
            cluster['cop_k_best_param']=cluster['cop_k_best'].params
            print('Summary_Best_Copula:', cluster['cop_k_best'].name, cluster['cop_k_best'].params)
    elif type(self.cop_type_opt)!=list:
        raise ValueError('Set valid value for cop_type_opt')
        
    return clusters


def initialize_clusters_random_parall(self, X):
    plot=True
    #step 0 con una iterata random
    
    n_features=self.n_features
    len_dataset=self.n_sample
    clusters=[]
    rdmclust= split_random(len_dataset, self.n_clusters)
    labels=np.empty(len_dataset)
    for k in range(self.n_clusters):
        labels[rdmclust[k]]=k
    for k in range(self.n_clusters):
        clusters.append({
        'marginals_': X[np.where(labels==k)],
        'mean_k': np.mean(X[np.where(labels==k)],axis=0), 
        'std_k':np.std(X[np.where(labels==k)],axis=0),  
        'pi_k': 1.0 / self.n_clusters,
        'al_k': np.mean(X[np.where(labels==k)],axis=0)*0+0.99,
        'bet_k': np.std(X[np.where(labels==k)],axis=0)*0+1e-10
       # 'ce_k': np.std(X[np.where(kmeans.labels_==k)],axis=0)*0+0.99
    })
    if type(self.cop_type_opt)==list:
        print('Fit copula step 0')
        print('first marginal inizialization of pdf and cdf with normal distribution')
        for cluster in clusters:
            #fit marginal for the copula
            mu_k = cluster['mean_k']
            std_k = cluster['std_k']
            normpdf= np.zeros(X.shape)
            #memorizzo l'empirica cumulata sulla base della inizializazione fatta con una normale
            #memorizzo la kernel density sulla base dell'inizializazione fatta con la normale
            #se si vuole la cdf calcolata con la normaledecommentare la parte gialla sotto
            
            norm_k_cdf= np.zeros(X.shape)
            for d in range (n_features):
                norm_k_cdf[:,d]=norm.cdf(X[:,d],cluster['mean_k'][d],cluster['std_k'][d])
                if plot:
                    plt.plot(np.sort(norm_k_cdf[:,d]))
                    plt.show()
            #this is to have not error in evaluation of cop_log_like
            
            cluster['marginal_cdf']=norm_k_cdf
            #this is to have not error in evaluation of cop_log_like
            cluster['marginal_cdf']=np.where(cluster['marginal_cdf']>=0.99, 1-1e-8,cluster['marginal_cdf'])
            cluster['marginal_cdf']=np.where(cluster['marginal_cdf']<=0.01, 1e-8,cluster['marginal_cdf'])
            # norm_k_cdf=np.round(norm_k_cdf,7)
            # norm_k_cdf=np.where(norm_k_cdf>=0.99, 1-1e-2,norm_k_cdf)
            # norm_k_cdf=np.where(norm_k_cdf<=0.1, 1e-2,norm_k_cdf)
            # cluster['marginal_cdf']=norm_k_cdf.copy()
            

            
            if self.maximization_type=='EMP_spline':
                #memorizzo l'empirica cumulata sulla base della inizializazione fatta con una normale
                cluster['emp_cdf']=cluster['marginal_cdf']
                #memorizzo la kernel density sulla base dell'inizializazione fatta con la normale
                for j in range(n_features):
                    normpdf[:,j]=norm.pdf(X[:,j], mu_k[j], std_k[j])
                cluster['emp_pdf']=np.copy(normpdf)
            
            #se si vuole la pdf calcolata con fftkde decommentare la parte gialla sotto
            '''
            emp_pdf= np.zeros(X.shape)
            kde=[]
            for d in range(self.n_features):
                
                kde_=FFTKDE(kernel='gaussian', bw="silverman").fit(cluster['marginals_'][:,d])#,weights= cluster['ecdf'][:,d])
                x,y=kde_.evaluate(2**10)
                kde.append(interp1d(x, y, fill_value=1.0e-6,bounds_error=False))
                emp_pdf[:,d]=kde[d](X[:,d])
            cluster['emp_pdf']=np.copy(emp_pdf)
            '''
    
            
            # fit copula
            
            cop_fam=cop_family_new(self.cop_type_opt)
            aic=[]
            bic=[]
            cluster['cop_k_list']=[]
            for name, cop in cop_fam:
                cop=cop(dim=self.n_features)
                #fit=fit_copula(cop, cluster['marginal_cdf'], x0=None, method='ml', optim_options={'method':'L-BFGS-B'}, verbose=1, scale=0.001)#optim_options={'method':'L-BFGS-B','options':{'ftol':1e-03, 'eps':1e-05,'maxfun':500, 'maxiter': 500,'gtol': 1.0e-3}}, verbose=1, scale=0.01)
                fit=fit_copula(cop, pobs(cluster['marginals_']), x0=None, method='ml', optim_options={'method':'L-BFGS-B'}, verbose=1, scale=0.001)#optim_options={'method':'L-BFGS-B','options':{'ftol':1e-03, 'eps':1e-05,'maxfun':500, 'maxiter': 500,'gtol': 1.0e-3}}, verbose=1, scale=0.01)
                #if math.isnan(model.params)==False:
                cluster['cop_k_list'].append(cop)
                #log_like=-(cop.log_lik(cluster['marginal_cdf'], to_pobs=False))
                #log_like=(cop.log_lik(pobs(cluster['marginals_']), to_pobs=True))
                log_like=fit.log_lik                  
                        
                aic_value=2*len(np.unique(cop.params))*len(cluster['marginals_'])-2*(log_like)
                bic_value=-2*(log_like)+len(np.unique(cop.params))*len(cluster['marginals_'])
                if math.isnan(aic_value)==False and math.isinf(aic_value)==False:
                    aic.append({'model':cop,
                                'aic':aic_value,
                                })
                    bic.append({'model':cop,
                                'bic':bic_value,
                                })
            best_aic = min(aic, key=lambda x: x['aic'])
            cluster['cop_k_best']=best_aic['model'] 
            cluster['cop_k_best_param']=cluster['cop_k_best'].params
            print('Summary_Best_Copula:', cluster['cop_k_best'].name, cluster['cop_k_best'].params)
    elif type(self.cop_type_opt)!=list:
        raise ValueError('Set valid value for cop_type_opt')
    
    clustersr = [] 
    slikr = []
    metricr=[]
    
    #step 1-nr random
    
    nr =10 # numero di iterate iniziali
    copMM_n = np.zeros( (self.n_sample,nr) )
    cc=clusters
    clustersr.append(cc)
    
    resp = expectation_step_EMP(self,X,clusters)
    
    opt_max_EMP=self.opt_max_EMP_spline
    #maximization_step_EMP(X, clusters, resp)
    self.clusters=clusters
    maximization_step_EMP_spline(self,X, clusters, resp, opt_max_EMP)
    
    copMM_n[:,0],_=self.predict(X)        
    slik,lik, met, met2 = eval_likelihood_EMP(self,X, clusters)
    
    clustersr.append(clusters)
    
    print('------------------------||||',slik) 
    slikr.append(slik)
    metricr.append(met)#aggiunto dopo
    for ii in range(nr-1): 
          self.init_clust='random'
          clusters = initialize_clusters_base(self,X)
          #clusters = initialize_clusters(X, self.n_features, self.n_cluster, self.init_clust, self.opt_maximization)
          resp = expectation_step_EMP(self,X, clusters)
          self.clusters=clusters
          #gau_copn[:,ii+1]=self.predict(X)  
          opt_max_EMP=self.opt_max_EMP_spline
          #maximization_step_IFM(self,X, clusters, resp)
          maximization_step_EMP_spline(self,X, clusters, resp, opt_max_EMP) 
          copMM_n[:,ii+1],_=self.predict(X)  
          slik,lik, met , met2 = eval_likelihood_EMP(self,X, clusters)
          print('------------------------||||',slik) 
          slikr.append(slik)
          metricr.append(met)
          
          clustersr.append(clusters)
    ind = np.argmax(np.abs(slikr))
    #ind=np.argmin(metricr)#aggiunto dopo
    print('max lik ind ', ind)# massima likelihood
    clusters = clustersr[ind]  
    kmeansdf = KMeans(self.n_clusters).fit(copMM_n)#.reshape(-1,1))
    gaudf= kmeansdf.labels_
    AD = np.zeros(nr)
    for i in range(nr):
        AD[i] = adjusted_mutual_info_score( np.reshape(gaudf,(X.shape[0])),np.array(np.reshape(copMM_n[:,i],(X.shape[0])),dtype=np.int32))
    ind = np.argmax(AD)      # massima mutual score
    print('max ams ind', ind)
    clusters = clustersr[ind]  
    self.clusters=clusters
    return clusters







