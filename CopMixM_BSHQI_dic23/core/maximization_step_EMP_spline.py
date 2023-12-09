"""

@author: Cristiano

MAXIMIZATION STEPS BASED ON EMPIRICAL MARGINAL ESTIMATION METHOD with SPLINE
"""
#----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
#from Utils.myestimator import fit_copula
from copulae.copula.estimator import fit_copula
from copulae.core import pseudo_obs
import scipy
from scipy.stats import norm
from scipy.optimize import OptimizeResult, minimize
from copulae.stats import multivariate_normal as mvn, norm
from KDEpy import FFTKDE
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import splrep, splder, splantider, splev, make_interp_spline
#from utils.distr_emp_bs import distempcont_bs2 as de
from utils.distr_emp_bs_23_new import BSemp as de

#from utils.emp_distribution import emp_distr
from utils.emp_distribution_23_new import emp_distr_new as emp_distr
from scipy.optimize import Bounds
from copulae.copula.estimator.misc import is_archimedean, is_elliptical
import math
from copulae.core import pseudo_obs as pobs

def maximization_step_EMP_spline(self,X, clusters,resp, opt_max_EMP):
    N = float(X.shape[0])
    n_features=X.shape[1]
    n_cluster=len(clusters)
    #resp=np.around(resp,8)
    #kde_pdf=np.zeros((X.shape[0] ,n_features))
    #ecdf=np.zeros((X.shape[0],n_features))
    index_cluster=np.argmax(resp, axis=1).reshape(-1,1)
    
    k=0
    for cluster in clusters:
        
        resp_nk = np.copy(cluster['resp_nk'])
        mu_k = cluster['mean_k']
        std_k = cluster['std_k']
        cop_k=cluster['cop_k_best']
        N_k = np.sum(resp_nk, axis=0)#controlla bene!!!
        print('component num:', k)
        print('number of element in cluster '+str(k), N_k,'\n parameter copula '+str(k), cop_k.params, '\ncopula name', cop_k.name)
        '''
        evaluation of cluster probability pi
        
        
        '''
        pi_k = N_k / N
        cluster['pi_k'] = pi_k
        
        
        cluster['marginals_']: X[np.where(index_cluster==k)]
        if self.init_clust=='hdbscan':
            bnds_mu=[(min(cluster['marginals_'][:,i]),max(cluster['marginals_'][:,i])) for i in range(n_features)]
            bnds_sd=[(1e-1,(max(cluster['marginals_'][:,i])-min(cluster['marginals_'][:,i]))/4) for i in range(n_features)]
        else:
            bnds_mu=[(min(cluster['marginals_'][:,i]),max(cluster['marginals_'][:,i])) for i in range(n_features)]
            bnds_sd=[(1e-1,(max(cluster['marginals_'][:,i])-min(cluster['marginals_'][:,i]))/4) for i in range(n_features)]
            bnds=bnds_mu+bnds_sd
            bnds_resp=[(1e-8,1-1e-8)]*len(resp_nk)

      #   bnds=[(1e-8,1e3) for i in range(n_features)]
      #  # bnds=bnds+bnds#+bnds
      # #  print('muk', mu_k,'stk',std_k)     
      # #  print('al_k',al_k,'bet_k',bet_k)
  
     
    
        
      #    # params=np.concatenate((al_k,bet_k,resp_nk),axis=0)
      #   res_EMP1=[]
      #   params=np.concatenate((al_k,bet_k,resp_nk),axis=0)
      #   res_EMP1 = minimize(fun_EMP1, params, args=(X,cop_k,resp_nk,index_cluster,k), method='L-BFGS-B', bounds=bnds)
      #   '''upload of mu_k and sd_k find before and stored in cluster'''
      #    #second step of ECM
      #   cluster['al_k']=res_EMP1['x'][0:n_features]
      #   cluster['bet_k']=res_EMP1['x'][n_features:2*n_features]
      #   cluster['ce_k']=res_EMP1['x'][2*n_features:]
      #   print('res optimization  1:', res_EMP1['x'])     
     
      #   al_k = np.copy(cluster['al_k'])
      #   bet_k = np.copy(cluster['bet_k'])
         # ce_k =     np.copy(cluster['ce_k'])
    
          
        (kde_pdf,ecdf)=emp_distr(X,resp_nk,index_cluster,k)
    
       
     #   for d in range(n_features): 
     #         kde_pdf[:,d]=norm.pdf(X[:,d],cluster['al_k'][d],cluster['bet_k'][d])
     #    
     #  for d in range(n_features):
     #       ecdf[:,d]=norm.cdf(X[:,d],cluster['al_k'][d],cluster['bet_k'][d])
            
            
    
        cluster['emp_cdf']=np.copy(ecdf)

        cluster['emp_pdf']=np.copy(kde_pdf)
        
        emp_cdf=cluster['emp_cdf']
        
        if len(self.cop_type_opt)==1 and self.cop_type_opt[0]!='all':
            if cop_k.name == 'Gaussian':
                param_cop=cop_k.params
                #(l,m)=cop_k._bounds
                bnds_cop=[(-1+1e-2,1-1e-2)]*(len(param_cop))
                #bnds_cop=Bounds(cop_k.bounds[0][0], cop_k.bounds[1])
                res_EMP = minimize(fun_EMP, param_cop, args=(emp_cdf, cop_k ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-05, 'maxfun': 150, 'maxiter': 150})
       
            elif cop_k.name == 'Student':
                param_cop=cop_k.params
                lower = np.array((2,-1+1.0e-6))
                upper = np.array((30,1-1.0e-6))
                bnds_cop=Bounds(lower, upper)
                #param_cop=([cop_k.params[0], cop_k.params[1][0]])
                res_EMP = minimize(fun_EMP, param_cop, args=(emp_cdf, cop_k ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 50, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-05, 'maxfun': 150, 'maxiter': 150})
    
            elif cop_k.name not in ['Gaussian','Student']:
                param_cop=cop_k.params
                bnds_cop=[(cop_k.bounds[0],cop_k.bounds[1])]
                res_EMP = minimize(fun_EMP, param_cop, args=(emp_cdf, cop_k ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 50, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-05, 'maxfun': 150, 'maxiter': 150})
     
                    #fit_copula_resp(cop, ecdf, responsability=resp_nk, x0=param_cop, method='ml',optim_options={'method':'L-BFGS-B'},verbose=1, scale=1)
            
    
            #res=fit_copula_resp(cop_k, cluster['marginal_cdf'], resp_nk, x0=param_cop, method='ml', optim_options={'method':'L-BFGS-B','options':{'ftol':1e-03, 'eps':1e-03,'maxfun':500, 'maxiter': 500,'gtol': 1.0e-3}}, verbose=1, scale=0.001)
            cop_k.params=res_EMP['x']
            print('res optimization :', res_EMP['x'])       
                  
            #update the params of copula
            log_like=-(cop_k.log_lik(emp_cdf, to_pobs=True))
            aic_value=-2*(log_like)+2*len(np.unique(cop_k.params))#*len(cluster['marginals_'])
            #stores the new copula with new params
            cluster['cop_k_best']=cop_k
            cluster['pi_k'] = pi_k
            cluster['aic']=aic_value
            print('component update num:', k)
            print('update the params of copula--->done---->',cop_k.name, cop_k.params)
        
        elif len(self.cop_type_opt)!=1 or self.cop_type_opt==['all']:
            aic=[]
            bic=[]
            for cop in cluster['cop_k_list']:
                #fit_copula(cop, ecdf, x0=cop.params, method='ml',verbose=1, optim_options= None, scale=0.01)
                #estimate=fit_copula_resp(cop, ecdf, responsability=resp_nk, x0=cop.params, method='ml',optim_options={'method':'L-BFGS-B','options':{'ftol':1e-03, 'eps':1e-05,'maxfun':500, 'maxiter': 500,'gtol': 1.0e-3}}, verbose=1, scale=0.01)
                #estimate=fit_copula_resp(cop, ecdf, responsability=resp_nk, x0=cop.params, method='ml',optim_options=None, verbose=1, scale=0.01)
                if cop.name == 'Gaussian':
                    param_cop=cop.params
                    
                    #(l,m)=gc_k._bounds
                    #bnds_cop=[[-1+20e-02,1-20e-02]]*len(param_cop)
                    bnds_cop=Bounds(cop.bounds[0][0], cop.bounds[1][0])
                    #fit=fit_copula(cop, cluster['marginal_cdf'], x0=None, method='ml', optim_options={'method':'L-BFGS-B','options':{'ftol':1e-05, 'eps':1e-05,'maxfun':100, 'maxiter': 500,'gtol': 1.0e-3}}, verbose=1, scale=0.1)
                    #fit=fit_copula(cop, emp_cdf, x0=None, method='ml', optim_options={'method':'L-BFGS-B','options':{'ftol':1e-05, 'eps':1e-05,'maxfun':100, 'maxiter': 500,'gtol': 1.0e-3}}, verbose=1, scale=0.1)
                    print('fit parameter',cop.params)
                    res_EMP = minimize(fun_EMP, param_cop, args=(emp_cdf, cop ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-05, 'eps': 1e-05, 'maxfun': 150, 'maxiter': 150})
                    print(res_EMP['x'])
                if cop.name == 'Student':
                    param_cop=cop.params
                    lower = np.array((2,-1+1.0e-6))
                    upper = np.array((20,1-1.0e-6))
                    bnds_cop=Bounds(lower, upper)
                #param_cop=([cop_k.params[0], cop_k.params[1][0]])
                    res_EMP = minimize(fun_EMP, param_cop, args=(emp_cdf, cop ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-05, 'maxfun': 150, 'maxiter': 150})
       
                elif cop.name not in ['Gaussian','Student']:
                    param_cop=cop.params
                    #param_cop=1
                    bnds_cop=[(cop.bounds[0],cop.bounds[1])]
                    res_EMP = minimize(fun_EMP, param_cop, args=(emp_cdf, cop ,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 100, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-05, 'maxfun': 150, 'maxiter': 150})
                    #fit_copula_resp(cop, ecdf, responsability=resp_nk,x0=cop.params, method='ml',optim_options={'method':'L-BFGS-B'}, verbose=1, scale=1)
                cop.params=res_EMP['x']
                print('cop_params',cop.params)
                #log_like=(cop.log_lik(cluster['marginal_cdf'], to_pobs=False))
                
                
                log_like=-(cop.log_lik(emp_cdf, to_pobs=True))
                # t1=cop.pdf(emp_cdf, log=True)
                # t2=resp_nk.T*t1
                # log_like=-np.log(np.sum(t2))
                
                aic_value=-2*(log_like)+2*len(np.unique(cop.params))#*len(cluster['marginals_'])
                #aic_value=-2*(log_like)*len(cluster['marginals_'])+2*len(np.unique(cop.params))
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
            cluster['aic']=best_aic['aic']
            cluster['cop_k_best_param']=cluster['cop_k_best'].params
            cluster['pi_k'] = pi_k
            print('component update num:', k)
            print('res optimization new copula:', cluster['cop_k_best'].name, cluster['cop_k_best_param']) 
        k=k+1
        # param_cop=cop_k.params
        # (l,m)=cop_k._bounds
        # bnds_cop=[(-1+1e-8,1-1e-8)]*(len(param_cop))
        # #-----------------------------------
        # #bnds=[bnds_cop,bnds_resp]
        # #param=np.concatenate(param_cop,bnds_resp)
        # #-----------------------------------
          
        # res_EMP=[]
        # res_EMP = minimize(fun_EMP, param_cop, args=(ecdf ,cop_k,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 10, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-02, 'maxfun': 150, 'maxiter': 150})
        # #res_EMP = minimize(fun_EMP, param, args=(ecdf ,gc_k,resp_nk), method='L-BFGS-B', bounds=bnds_cop, options={'maxcor': 10, 'ftol': 1e-05, 'gtol': 1e-02, 'eps': 1e-02, 'maxfun': 150, 'maxiter': 150})
        # cop_k.params=res_EMP['x']
        # print('res optimization :', res_EMP['x'])     
        # print('cluster_num', k)
        # '''
        # print('-----------fit copula----------------')
        # fit_copula(gc_k,cluster['emp_cdf'], x0=None, method='ml', optim_options={'method':'SLSQP'}, verbose=1, scale=0.001)
        # '''
        
        # cluster['cop_k_best']=cop_k
        # cluster['pi_k'] = pi_k
        # print('update the params of copula--->done---->',cop_k.params)

               

'''
------------------------------------------------------------------------------
'''


def fun_EMP2(params, X, cop_k, resp_nk,index_cluster,k):
    n_features=X.shape[1]
    al_k=params[:n_features]
    bet_k = params[n_features:]        
    log_k_pdf= np.zeros(X.shape)
    k_cdf= np.zeros(X.shape)
    for j in range(n_features):   
        k_cdf[:,j]=norm.cdf(X[:,j], al_k[j],bet_k[j])
        log_k_pdf[:,j]=norm.pdf(X[:,j], al_k[j],bet_k[j])
        log_k_pdf[:,j]=np.log(1e-8+log_k_pdf[:,j])
        
        
    k_cdf=np.around(k_cdf,8)
    k_cdf=np.where(k_cdf>=1-1e-8, 1-1e-8,k_cdf)
    k_cdf=np.where(k_cdf<=1e-8, 1e-8,k_cdf)
    return -np.sum(resp_nk.T*(cop_k.pdf(k_cdf,log=True)+np.sum(log_k_pdf,axis=1)))

def fun_EMP1(params, X, cop_k, resp_nk,index_cluster,k):
    n_features=X.shape[1]      
    log_k_pdf= np.zeros(X.shape)
    k_cdf= np.zeros(X.shape)  
    for j in range(n_features):   
        log_k_pdf[:,j]=np.where(log_k_pdf[:,j]<=1e-8, 1e-8,log_k_pdf[:,j])
        log_k_pdf[:,j]=np.log(log_k_pdf[:,j])
        
    return -np.sum(resp_nk.T*(cop_k.pdf(k_cdf,log=True)+np.sum(log_k_pdf,axis=1)))
     

'''
-------------------------------------------------------------------------------
'''

def fun_EMP(param_cop, emp_cdf, cop_k, resp_nk):
        cop_k.params=param_cop
        if is_elliptical(cop_k):
            cop_k._force_psd()
            t1=cop_k.pdf(emp_cdf, log=True)
            t2=np.dot(resp_nk,t1)
            return -np.sum(t2)
        elif is_archimedean(cop_k):
            t1=cop_k.pdf(emp_cdf, log=True)
            t2=np.dot(resp_nk,t1)
            return -np.sum(t2)

def fun_EMP_gau(param_cop, ecdf, gc_k, resp_nk):
        gc_k.params=param_cop
        gc_k._force_psd()
        q = norm.ppf(ecdf)
        sigma=gc_k.sigma
        d = mvn.logpdf(q, cov=sigma) - norm.logpdf(q).sum(1)
        return -np.sum( resp_nk.T*d)
    


