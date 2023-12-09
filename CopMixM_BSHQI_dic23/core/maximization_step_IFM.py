'''
MAXIMIZATION STEPS BASED ON IFM METHOD
---------------
'''


import numpy as np
from copulae.core import pseudo_obs as pobs
import scipy
from scipy.stats import norm
from scipy.optimize import OptimizeResult, minimize
from copulae.stats import multivariate_normal as mvn, norm
#from copulae.copula.estimator import max_loglike_cop_resp
#from copulae.copula.estimator.estimator_cop_resp import fit_copula_resp
from utils.cop_family import cop_family_new
from copulae.copula.estimator import fit_copula
#from copulae.copula.estimator import fit_copula_resp
from scipy.optimize import Bounds
from copulae.copula.estimator.misc import is_archimedean, is_elliptical
import math

def maximization_step_IFM(self,X, clusters,resp):
    N = self.n_sample
    n_features=self.n_features
    n_cluster=self.n_clusters
    #resp=np.around(resp,5)
    index_cluster=np.argmax(resp, axis=1).reshape(-1,1)

    
    for k, cluster in enumerate(clusters):
        
        resp_nk = cluster['resp_nk']
        mu_k = cluster['mean_k']
        cop_k=cluster['cop_k_best']
        norm_k_cdf=cluster['marginal_cdf']
        N_k = np.sum(resp_nk, axis=0)#controlla bene!!!
        print(cop_k.params)
        '''
        evaluation of cluster probability pi
        '''
        pi_k = N_k / N
        
        cov_k = np.zeros((X.shape[1], X.shape[1]))
        resp_nk=resp_nk.reshape(-1,1)
        mu_k = np.sum(np.dot(resp_nk.T,X), axis=0)/ N_k
        
        for j in range(N):
            diff = (X[j] - mu_k).reshape(-1, 1)
            cov_k =cov_k + resp_nk[j] * np.dot(diff, diff.T)
        cov_k =cov_k / N_k

        cluster['mean_k'] = mu_k
        cluster['std_k'] =np.sqrt(np.diag(cov_k))
     

        for d in range (n_features):
            norm_k_cdf[:,d]=norm.cdf(X[:,d],cluster['mean_k'][d],cluster['std_k'][d])
        #norm_k_cdf=np.around(norm_k_cdf,8)
        norm_k_cdf=np.where(norm_k_cdf>=1-1e-8, 1-1e-8,norm_k_cdf)
        norm_k_cdf=np.where(norm_k_cdf<=1e-8, 1e-8,norm_k_cdf)
        cluster['marginal_cdf']=norm_k_cdf
        
        if len(self.cop_type_opt)==1 and self.cop_type_opt[0]!='all':
            #cluster['marginals_']: X[np.where(index_cluster==k)]
            if cop_k.name == 'Gaussian':
                param_cop=cop_k.params
                #bnds_cop=[[-1+1e-08,1-1e-08]]*len(param_cop)
                bnds_cop=Bounds(cop_k.bounds[0][0], cop_k.bounds[1][0])
                res_IFM = minimize(fun_IFM2, param_cop, args=(norm_k_cdf, cop_k ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-05, 'eps': 1e-05, 'maxfun': 150, 'maxiter': 150})
   
            elif cop_k.name == 'Student':
                param_cop=cop_k.params
                lower = np.array((2,-1+1.0e-6))
                upper = np.array((30,1-1.0e-6))
                bnds_cop=Bounds(lower, upper)
                #param_cop=([cop_k.params[0], cop_k.params[1][0]])
                res_IFM = minimize(fun_IFM, param_cop, args=(norm_k_cdf, cop_k ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-02, 'maxfun': 150, 'maxiter': 150})
    
            elif cop_k.name not in ['Gaussian','Student']:
                param_cop=cop_k.params
                bnds_cop=[(cop_k.bounds[0],cop_k.bounds[1])]
                res_IFM = minimize(fun_IFM, param_cop, args=(norm_k_cdf, cop_k ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-02, 'maxfun': 150, 'maxiter': 150})
     
                    #fit_copula_resp(cop, ecdf, responsability=resp_nk, x0=param_cop, method='ml',optim_options={'method':'L-BFGS-B'},verbose=1, scale=1)
            

            #res=fit_copula_resp(cop_k, cluster['marginal_cdf'], resp_nk, x0=param_cop, method='ml', optim_options={'method':'L-BFGS-B','options':{'ftol':1e-03, 'eps':1e-03,'maxfun':500, 'maxiter': 500,'gtol': 1.0e-3}}, verbose=1, scale=0.001)
            cop_k.params=res_IFM['x']
            print('res optimization :', res_IFM['x'])       
                  
            #update the params of copula
            
            #stores the new copula with new params
            cluster['cop_k_best']=cop_k
            cluster['cop_k_best_param']=cop_k
            #cluster['pi_k'] = pi_k
            print('component num:', k)
            print('update the params of copula--->done---->',cop_k.params)
        elif len(self.cop_type_opt)!=1 or self.cop_type_opt==['all']:
            aic=[]
            for cop in cluster['cop_k_list']:
                #fit_copula(cop, ecdf, x0=cop.params, method='ml',verbose=1, optim_options= None, scale=0.01)
                #estimate=fit_copula_resp(cop, ecdf, responsability=resp_nk, x0=cop.params, method='ml',optim_options={'method':'L-BFGS-B','options':{'ftol':1e-03, 'eps':1e-05,'maxfun':500, 'maxiter': 500,'gtol': 1.0e-3}}, verbose=1, scale=0.01)
                #estimate=fit_copula_resp(cop, ecdf, responsability=resp_nk, x0=cop.params, method='ml',optim_options=None, verbose=1, scale=0.01)
                if cop.name == 'Gaussian':
                    param_cop=cop.params
                    if np.isnan(param_cop):
                        pass
                    
                    #(l,m)=gc_k._bounds
                    #bnds_cop=[[-1+20e-02,1-20e-02]]*len(param_cop)
                    bnds_cop=Bounds(cop.bounds[0][0], cop.bounds[1][0])
                    res_IFM = minimize(fun_IFM, param_cop, args=(norm_k_cdf, cop ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-02, 'maxfun': 150, 'maxiter': 150})
                 
                if cop.name == 'Student':
                    param_cop=cop.params
                    if np.isnan(param_cop[0]) or np.isnan(param_cop[0]):
                        pass
                    lower = np.array((2,-1+1.0e-6))
                    upper = np.array((30,1-1.0e-6))
                    bnds_cop=Bounds(lower, upper)
                #param_cop=([cop_k.params[0], cop_k.params[1][0]])
                    res_IFM = minimize(fun_IFM, param_cop, args=(norm_k_cdf, cop ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-02, 'maxfun': 150, 'maxiter': 150})
       
                elif cop.name not in ['Gaussian','Student']:
                    param_cop=cop.params
                    if np.isnan(param_cop):
                        pass
                    bnds_cop=[(cop.bounds[0],cop.bounds[1])]
                    res_IFM = minimize(fun_IFM, param_cop, args=(norm_k_cdf, cop ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-02, 'maxfun': 150, 'maxiter': 150})
                    cop.params=res_IFM['x']
                    #fit_copula_resp(cop, ecdf, responsability=resp_nk,x0=cop.params, method='ml',optim_options={'method':'L-BFGS-B'}, verbose=1, scale=1)
                cop.params=res_IFM['x']
                log_like=-(cop.log_lik(cluster['marginal_cdf'], to_pobs=False))
                
                if cop.name=='Student':
                    aic.append({'model':cop,
                                'aic':4+2*(log_like),
                                })
                else:
                    aic_value=2+2*(log_like)
                    if math.isnan(aic_value)==False and math.isinf(aic_value)==False:
                        aic.append({'model':cop,
                                    'aic':aic_value,
                                    })
            best_aic = min(aic, key=lambda x: x['aic'])
            cluster['cop_k_best']=best_aic['model']
            cluster['aic']=best_aic['aic']
            cluster['cop_k_best_param']=cluster['cop_k_best'].params
        cluster['pi_k'] = pi_k
        print('component num:', k)
        print('res optimization new copula:', cluster['cop_k_best'].name, cluster['cop_k_best_param']) 

        
            
    
#-------------------------------------------------------------------------------

def fun_IFM(param_cop, norm_k_cdf, cop_k, resp_nk):
        cop_k.params=param_cop
        if is_elliptical(cop_k):
            cop_k._force_psd()
            t1=cop_k.pdf(norm_k_cdf, log=True)
            t2=resp_nk.T*t1
            return -np.sum(t2)
        elif is_archimedean(cop_k):
            t1=cop_k.pdf(norm_k_cdf, log=True)
            t2=resp_nk.T*t1
            return -np.sum(t2)
        
#------------other calculation of function IFM---------------------------------------------
def fun_IFM1(param_cop, norm_k_cdf, cop_k, resp_nk):
        cop_k.params=param_cop
        q = norm.ppf(norm_k_cdf)
        sigma=cop_k.sigma
        try:
            d = mvn.logpdf(q, cov=sigma) - norm.logpdf(q).sum(1)
        except:
            cop_k._force_psd()
            d = mvn.logpdf(q, cov=sigma) - norm.logpdf(q).sum(1)
        return -np.sum(resp_nk.T*d)

def fun_IFM2(param_cop, norm_k_cdf, cop_k, resp_nk):
        cop_k.params=param_cop
        cop_k._force_psd()
        q = norm.ppf(norm_k_cdf)
        sigma=cop_k.sigma
        d = mvn.logpdf(q, cov=sigma) - norm.logpdf(q).sum(1)
        return -np.sum(resp_nk.T*d)