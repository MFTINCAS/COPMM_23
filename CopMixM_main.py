"""
@author: Cristiano
"""

from pathlib import Path
import numpy as np
from data_utils.generate_dataset import generate_dataset
from core import CopMixM
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
from utils import metrics_result
from copulae.core import pseudo_obs as pobs
from statsmodels.nonparametric.bandwidths import bw_silverman
from core.CopMixM import CopMixM
import pyvinecopulib as pv
from sklearn.metrics import f1_score, confusion_matrix
import csv
from sklearn.cluster import KMeans
from sklearn import datasets
import random
from sklearn import metrics
from sklearn.metrics import accuracy_score
from utils.mytools import reconstructed_svd 

plt.style.use('ggplot')
plt.style.use('default')

path = Path('C:/Users/Cristiano/Desktop/Python_Code')

'''
Choose the datasets.
synt_1, synt_2, synt_3, synt_4, iris, ais, cancer, protein,
burned, harmiston, sanfrancisco, santabarbara.
'''


X, y=generate_dataset()

X = pd.read_csv(path/'CopMixM/CopMM_23_class/Datasets_CopMixM/ais.csv', header=None)
X=X.values
#gt = pd.read_csv(path/'CopMixM/CopMM_23_class/Datasets_CopMixM/gt_ais.csv', header=None)
#gt=np.array(gt)
#X_svd, recostructed=reconstructed_svd(X, 3)
#X=X_svd
X=np.array(X)

gt=d_vineexamplecsv[:,3]

X=X[:,:2]
gt=aa[:,2]
plt.scatter(X[:,0],X[:,1],s=3, c=gt)

'''
if consider all copula write cop_fam=['all']
if cop_fam is not all and you want one ore 2 or three or four copula then
pass cop_fam as list i.e. cop_fam=['Gaussian'], or cop_fam=['Gaussian', 'Clayton']
'''

'''
Initialize cluster option: 'k_means','random','random_parall'

for maximization the option is: 'IFM', 'EMP_spline'
'''


n_cluster=3
n_iter=150
tol=1e-4
init_clust='random'
parallel_iter=4 #solo se si sceglie di parallelizzare con inizializzazione random

random_iter=100

#cop_type_opt=['all']
cop_type_opt=['all']

'''
Maximization_type
EMP_spline
ECM
IFM
IFM_all
'''
maximization_type='EMP_spline'

opt_max_EMP_KDEpy={'kernel':'gaussian',
                   'bw':'silverman'
                   }

opt_max_EMP_spline={'emp_distr_linear':False,
                    'emp_distr_allbs':True, 
                    'distr_emp_bs2_bw':np.nan,
                    #'distr_emp_bs2_bw':'ISJ',
                    'distemp_bs2_linear':False
                    }

if init_clust!='random_parall':
    copMM=CopMixM(n_cluster,n_iter, tol, init_clust,cop_type_opt, 
                  maximization_type, opt_max_EMP_KDEpy, opt_max_EMP_spline).fit(X)
    cop_labels, clusters=copMM.predict((X))
    
    
    gmm = GaussianMixture(n_cluster,tol=1e-4, max_iter=500, covariance_type='diag',init_params='random').fit(X)
    
    
    
    aa=gmm.predict_proba((X))
    cc=np.argmax(aa,1)
    bb=np.where(np.abs(aa[:,0]-aa[:,1])<5e-1)[0]
    for i in bb:
        cc[i]=random.choice([0,1])
        

    
    gmm=gmm.predict(X).reshape(-1,1)
    plt.scatter(X[:, 0], X[:, 1],c=cop_labels ,s=3, cmap='viridis', zorder=1)
    plt.title('cop_mixm')
    plt.show()
    
    plt.scatter(pobs(X[:, 0]), pobs(X[:, 1]),c=cop_labels ,s=3, cmap='viridis', zorder=1)
    plt.show()
    
    plt.scatter(X[:, 0], X[:, 1],c=gmm ,s=5, cmap='viridis', zorder=1)
    plt.title('gmm')
    plt.show()
    
    plt.scatter(X[:, 0], X[:, 1],c=cc ,s=5, cmap='viridis', zorder=1)
    plt.title('gmm corrected')
    plt.show()

    
    #plt.scatter(X[:, 0], X[:, 1],c=gt ,s=5, cmap='viridis', zorder=1)
    #	plt.title('gt')
    plt.show()
    Sil = metrics.silhouette_score(X, cop_labels)
    CH = metrics.calinski_harabasz_score(X, cop_labels)
    DB = metrics.davies_bouldin_score(X, cop_labels)
    print('Metrics for copula:\n'+'Sil:',Sil,'\nCH:',CH,'\nDB:',DB)
    Sil = metrics.silhouette_score(X, gmm.ravel())
    CH = metrics.calinski_harabasz_score(X, gmm.ravel())
    DB = metrics.davies_bouldin_score(X, gmm.ravel())
    print('Metrics for gmm:\n'+'Sil:',Sil,'\nCH:',CH,'\nDB:',DB)
    try:
        my_accuracy = accuracy_score(gt, cop_labels, normalize=False) / float(cop_labels.size)
        print('Misclassification_rate copmm:\n',my_accuracy)
        my_accuracy = accuracy_score(gt, gmm, normalize=False) / float(cop_labels.size)
        print('Misclassification_rate gmm:\n',my_accuracy)
    except NameError:
        gt=None

else:
    copMM_n=np.zeros( (X.shape[0],parallel_iter))
    
    for ns in range(parallel_iter):   
       copMM=CopMixM(n_cluster, n_iter, tol, init_clust, cop_type_opt,
                       maximization_type,opt_max_EMP_KDEpy, opt_max_EMP_spline).fit(X)
       copMM_n[:,ns],_=copMM.predict(X)
       #plt.scatter(pobs(X[:, 0]), pobs(X[:, 1]),c=gau_copn[:,ns],s=5, cmap='viridis', zorder=1)
       #plt.title('gau_cop i pseudo')
       #plt.show()
       plt.scatter((X[:, 0]), (X[:, 1]),c=copMM_n[:,ns],s=5, cmap='viridis', zorder=1)
       plt.title('gau_cop parallel iter n: ' +  str(ns))
       plt.show()
      
 
    
    kmeansdf = KMeans(n_cluster).fit(copMM_n)#.reshape(-1,1))
    gaudf= kmeansdf.labels_
    
    
    #confronto con gmm
    
    gmm = GaussianMixture(n_components=n_cluster,tol=1e-4, max_iter=500).fit(X)
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
      AD[i] = adjusted_mutual_info_score( np.reshape(gaudf,(X.shape[0])),np.array(np.reshape(copMM_n[:,i],(X.shape[0])),dtype=np.int32))
    print(AD)
      
    ind = np.argmax(AD)  
    # plt.scatter(pseudo_obs(X[:, 0]), pseudo_obs(X[:, 1]),c=gau_copn[:,ind],s=5, cmap='viridis', zorder=1)
    # plt.title('gau_cop best pseudo')
    # plt.show()
    plt.scatter((X[:, 0]), (X[:, 1]),c=copMM_n[:,ind],s=5, cmap='viridis', zorder=1)
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
    
    print("AD", AD)
    '''
    #decomment if gt exist
    print("ADgt")
    print(ADgt)
    print("ADgtgmm")
    print(ADgtgmm)
    '''
  


