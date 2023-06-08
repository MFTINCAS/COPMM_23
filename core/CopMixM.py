'''
----------------CopMM class algorithm----------------------------

@author: Cristiano

'''
import numpy as np
import copulae
from sklearn.mixture import GaussianMixture
from copulae.elliptical import GaussianCopula, StudentCopula
from copulae.archimedean import FrankCopula, ClaytonCopula, GumbelCopula
from utils import *
import pandas as pd
from utils.mytools import generate_dataset, result_table, test_dataset_auto, clean_map
from scipy.stats import norm
from core.initialize_clusters import initialize_clusters_base, initialize_clusters_random_parall
from core.maximization_step_IFM import maximization_step_IFM
from core.maximization_step_ECM import maximization_step_ECM
from core.maximization_step_EMP_spline import maximization_step_EMP_spline
from core.maximization_step_EMP_KDEpy import maximization_step_EMP_KDEpy
from core.eval_likelihood_EMP import eval_likelihood_EMP
from core.expectation_step_EMP import expectation_step_EMP
from core.expectation_step_IFM import expectation_step_IFM
from core.maximization_step_IFM_all import maximization_step_IFM_all
from core.expectation_step_ECM import expectation_step_ECM
from core.eval_likelihood import eval_likelihood
from scipy.stats import norm
from os.path import dirname, join as pjoin
from pathlib import Path
from utils import metrics_result
from os import makedirs, remove
from os.path import isfile, join, exists
from os import listdir
import shutil
import matplotlib.pyplot as plt
from sklearn import metrics

import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
robjects.numpy2ri.activate()

base = importr('base')
rvinecop = importr('rvinecopulib')
rcop=importr('copula')


class CopMixM:
    """ Copula Mixture Model
    
    Parameters
    -----------
        k: int , number of copula distributions
        
        seed: int, will be randomly set if None
        
        max_iter: int, number of iterations to run algorithm, default: 200
        
        if copula is not Gaussian set the name of copula from:
            'Student','Clayton', 'Frank', 'Gumbel'
            
            if marginals is not normal set 'empirical'
            
            default is: 
                
            copula_type='Gaussian', marginal_type='empirical', maximization_type= 'ECM'
       
    """
    def __init__(self, n_clusters, n_iter, tol=1e-4, init_clust='k-mean', cop_type_opt=['Gaussian'],maximization_type=None, opt_max_EMP_KDEpy=None, opt_max_EMP_spline=None, **args):
        
        

        self.n_clusters = n_clusters # number of Copula/clusters
        self.n_iter = n_iter
        self.tol=tol
        self.init_clust=init_clust
        self.cop_type_opt=cop_type_opt
        self.maximization_type=maximization_type
        self.opt_max_EMP_KDEpy=opt_max_EMP_KDEpy
        self.opt_max_EMP_spline=opt_max_EMP_spline
        
        
        
    
    def get_params(self):
        return (self.pi, self.copula, self.marginal)
    
    def fit(self, X):
        """Compute the Expectation-step and Maximization-step
        
        Parameters:
        -----------
        X: (N x D), data 
        
        Returns:
        ----------
        instance of Cop_MixM
        
        """
        
        N, D = X.shape
        
        self.n_sample=N
        
        self.n_features=D
        
        if self.init_clust=='random_parall':
            clusters=initialize_clusters_random_parall(self,X)
        else:
            clusters=initialize_clusters_base(self,X)
               
        scores = np.zeros((X.shape[0], self.n_clusters))
        scores_prob = np.zeros((X.shape[0], self.n_clusters))
        likelihoods =[]
        metrics=[]
        metrics2=[]
        tol=self.tol
        for i in range(self.n_iter):
            self.temp_iter=i
            if i==0:
                print('First Step after initialization:')
            

            if self.maximization_type=='IFM':
                resp = expectation_step_IFM(self, X, clusters)
                maximization_step_IFM(self,X, clusters, resp)
                likelihood_sum, likelihood,metric, metric2 = eval_likelihood(X, clusters)
                #likelihood, sample_likelihoods = get_likelihood(X, clusters)
                likelihoods.append(likelihood_sum)
                metrics.append(metric)
                metrics2.append(metric2)

                

            elif self.maximization_type=='EMP_spline':
                resp = expectation_step_EMP(self,X, clusters)
                self.resp=resp
                opt_max_EMP=self.opt_max_EMP_spline
                maximization_step_EMP_spline(self, X, clusters, resp, opt_max_EMP)
                
                likelihood_sum, likelihood , metric, metric2= eval_likelihood_EMP(self, X, clusters)
                #likelihood, sample_likelihoods = get_likelihood(X, clusters)
                metrics.append(metric)
                metrics2.append(metric2)
                likelihoods.append(likelihood_sum)
                
            if i>=1:
                print('Iter_n: ',i)
                
                if  np.abs((likelihoods[i]- likelihoods[i-1]))/(1+np.abs(likelihoods[i]))<tol:
                    lik_cop=likelihoods[i]
                    print('Iter_n: ',i,  '\nMax_likelihoods: ', lik_cop)
                    
                    break
                elif i==self.n_iter-1:
                    lik_cop=likelihoods[i]
            elif i==0:
                print('---------->Iter_n: ', i)
                '''
            if i>=1:
                if  np.abs((likelihoods[i]- likelihoods[i-1]))<tol:
                    break
                '''
        
        self.clusters=clusters
        
        
        
        print('Iter_n: ',i+1,  '\nMax_likelihoods: ', lik_cop)
        for i, cluster in enumerate(clusters): #ritorna l'indice di ogni cluster 
            scores[:, i] = np.log(cluster['resp_nk']).reshape(-1)
            scores_prob[:,i]=cluster['resp_nk'].reshape(-1)
        plt.figure(figsize=(10, 10))
        plt.title('Log-Likelihood')
        plt.plot(np.arange(1, len(likelihoods)+1), likelihoods)
        plt.show()
        
        '''ATTENTION!!!!
        if make np.argmax of scores you will obtain the same result of
        predict????'''
        
        self.scores=scores
        self.scores_prob=scores_prob
        self.likelihoods=likelihoods
        self.likelihood=likelihood
        self.metric=metrics
        self.metrics2=metrics2
        final_table=[]
        for k in range(self.n_clusters):
            if 'aic' in clusters[k]:
                final_table.append({'cluster':k, 'copula':clusters[k]['cop_k_best'].name,'aic':clusters[k]['aic'] })
            else:
                final_table.append({'cluster':k, 'copula':clusters[k]['cop_k_best'].name})
        print(pd.DataFrame(final_table))
        return self   
    
    def predict(self, X):
        """Returns predicted labels using Bayes Rule to
        Calculate the posterior distribution
        
        Parameters:
        -------------
        X: N*D numpy array
        
        Returns:
        ----------
        labels: predicted cluster based on 
        highest responsibility gamma.
        
        """
        labels = np.zeros((X.shape[0], self.n_clusters))
        clusters=self.clusters
        normpdf = np.zeros((X.shape))
        #full non parametric estimation of marginals
        if self.maximization_type=='IFM':
            for k, cluster in enumerate(clusters):
                pi_k = cluster['pi_k']
                mu_k = cluster['mean_k']
                std_k = cluster['std_k']
                cop_k=cluster['cop_k_best']
                norm_k_cdf=cluster['marginal_cdf']
                for d in range(self.n_features):
                    normpdf[:,d]=norm.pdf(X[:,d], mu_k[d], std_k[d])
                normpdf_mult=np.prod(normpdf,axis=1)    
                cop_pdf_k=cop_k.pdf(norm_k_cdf)
                cluster['cop_pdf_k']=cop_pdf_k
                #mvdgc='multi variate distribution gaussian copula'
                mvdgc = cop_pdf_k*normpdf_mult
                labels[:,k] = pi_k * mvdgc
            labels  = labels.argmax(1)
        # for k, cluster in enumerate(clusters):
        #     pi_k = cluster['pi_k']
        #     # if self.cop_fam=='all':
        #     #     cop_k=cluster['cop_k_best']
        #     # else:
        #     #     cop_k=cluster['cop_k']
        #     cop_k=cluster['cop_k_best']
        #     ecdf=cluster['marginal_cdf']
        #     al_k = cluster['al_k']
        #     bet_k = cluster['bet_k']
        #     kde_k_pdf=cluster['marginal_pdf']
        #     kdepdf_mult=np.prod(kde_k_pdf,axis=1)
        #     cop_pdf_k=cop_k.pdf(ecdf)
        #     cluster['cop_pdf_k']=cop_pdf_k
        #     #mvdgc='multi variate distribution gaussian copula'
        #     mvdgc = cop_pdf_k*kdepdf_mult
        #     labels[:,k] = pi_k * mvdgc
        # labels  = labels.argmax(1)
        elif self.maximization_type=='EMP_spline':
            for k, cluster in enumerate(clusters):
                pi_k = cluster['pi_k']
                ecdf=cluster['emp_cdf']
                al_k = cluster['al_k']
                bet_k = cluster['bet_k']
                kde_k_pdf=cluster['emp_pdf']
                cop_k=cluster['cop_k_best']
                kdepdf_mult=np.prod(kde_k_pdf,axis=1)
                cop_pdf_k=cop_k.pdf(ecdf)
                cluster['cop_pdf_k']=cop_pdf_k
                #mvdgc='multi variate distribution gaussian copula'
                mvdgc = cop_pdf_k*kdepdf_mult
                labels[:,k] = pi_k * mvdgc
            labels  = labels.argmax(1)
        self.labels_=labels
        return labels, clusters
            


    
    def predict_proba(self, X):
        """Returns posterior probability
        
        Parameters:
        -------------
        X: N*d numpy array
        
        Returns:
        ----------
        labels: predicted cluster based on 
        highest responsibility gamma.
        
        """
        post_proba = np.zeros((X.shape[0], self.n_clusters))
        normpdf = np.zeros((X.shape))

        clusters=self.clusters


        for k, cluster in enumerate(clusters):
            pi_k = cluster['pi_k']
            ecdf=cluster['emp_cdf']
            kde_k_pdf=cluster['emp_pdf']
            cop_k=cluster['cop_k_best']
            kdepdf_mult=np.prod(kde_k_pdf,axis=1)
            cop_pdf_k=cop_k.pdf(ecdf)
            cluster['cop_pdf_k']=cop_pdf_k
            #mvdgc='multi variate distribution gaussian copula'
            mvdgc = cop_pdf_k*kdepdf_mult
            post_proba[:,k] = (pi_k * mvdgc)# aggiustare questa
                
        return post_proba
    
    '''
    Opzione con parallelizazione
    '''   
    def parall_CopMixM(self, parallel_iter,X):
        gau_copn=np.zeros( (X.shape[0],parallel_iter) )
        for ns in range(parallel_iter):   
           gau_cop=CopMixM(n_cluster,200, tol=1e-4, init_clust='random', opt_maximization= 'EMP_new01',opt_max_EMP={'emp_distr_linear':False,'emp_distr_allbs':True, 'distr_emp_bs2_bw':np.nan,'distemp_bs2_linear':True}).fit(X)
           #gau_cop=CopGauMixM(n_cluster,100, tol=1e-4, init_clust='random').fit(X)
           gau_copn[:,ns]=gau_cop.predict(X)
           #plt.scatter(pseudo_obs(X[:, 0]), pseudo_obs(X[:, 1]),c=gau_copn[:,ns],s=5, cmap='viridis', zorder=1)
           #plt.title('gau_cop i pseudo')
           #plt.show()
           plt.scatter((X[:, 0]), (X[:, 1]),c=gau_copn[:,ns],s=5, cmap='viridis', zorder=1)
           plt.title('gau_cop parallel iter n: ' +  str(ns))
           plt.show()
     
        
        kmeansdf = KMeans(n_cluster).fit(gau_copn)#.reshape(-1,1))
        gaudf= kmeansdf.labels_
        
        
        #confronto con gmm
        
        gmm = GaussianMixture(n_components=n_cluster,tol=1e-4, max_iter=500, covariance_type='full', init_params='kmeans').fit(X)
        gmm=gmm.predict(X).reshape(-1,1)
        
        # plt.scatter(pseudo_obs(X[:, 0]), pseudo_obs(X[:, 1]),c=gaudf,s=5, cmap='viridis', zorder=1)
        # plt.title('gau_cop cum pseudo')
        # plt.show()
        plt.scatter((X[:, 0]), (X[:, 1]),c=gaudf,s=5, cmap='viridis', zorder=1)
        plt.title('gau_cop cum')
        plt.show()
        
        from sklearn.metrics.cluster import adjusted_mutual_info_score
        AD = np.zeros(parallel_iter)
        for i in range(parallel_iter):
          AD[i] = adjusted_mutual_info_score( np.reshape(gaudf,(X.shape[0])),np.array(np.reshape(gau_copn[:,i],(X.shape[0])),dtype=np.int32))
     
          
        ind = np.argmax(AD)  
        # plt.scatter(pseudo_obs(X[:, 0]), pseudo_obs(X[:, 1]),c=gau_copn[:,ind],s=5, cmap='viridis', zorder=1)
        # plt.title('gau_cop best pseudo')
        # plt.show()
        plt.scatter((X[:, 0]), (X[:, 1]),c=gau_copn[:,ind],s=5, cmap='viridis', zorder=1)
        plt.title('gau_cop best')
        plt.show()
        '''
        decomment if gt exist
        
        ADgt = np.zeros(parallel_iter)
        for i in range(parallel_iter):
          ADgt[i] = adjusted_mutual_info_score( np.reshape(gt,(X.shape[0])),np.array(np.reshape(gau_copn[:,i],(X.shape[0])),dtype=np.int32))
    
        ADgtgmm = adjusted_mutual_info_score( np.reshape(gt,(X.shape[0])),np.array(np.reshape(gmm,(X.shape[0])),dtype=np.int32))
        '''
        
        plt.scatter(X[:, 0], X[:, 1],c=gmm ,s=5, cmap='viridis', zorder=1)
        plt.title('gmm')
        plt.show()
        # plt.scatter(X[:, 0], X[:, 1],c=gt ,s=5, cmap='viridis', zorder=1)
        # plt.title('gt')
        # plt.show()
        plt.style.use('default')
        
        print("AD")
        print(AD)
        '''
        #decomment if gt exist
        print("ADgt")
        print(ADgt)
        print("ADgtgmm")
        print(ADgtgmm)
        '''
      
    
    
#if __name__ == "__main__":
     
    
    
    
