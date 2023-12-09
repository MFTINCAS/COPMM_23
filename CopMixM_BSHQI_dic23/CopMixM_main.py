"""
@author: Cristiano

Ultima versione per articolo Dicembre 2023
"""


from pathlib import Path
import numpy as np
from data_utils.generate_dataset import generate_dataset
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
from core.CopMixM import CopMixM
from sklearn.metrics import f1_score, confusion_matrix
import csv
from sklearn.cluster import KMeans
from sklearn import datasets
import random
from sklearn import metrics
from sklearn.metrics import accuracy_score
from utils.mytools import reconstructed_svd 
from utils.choose_dataset import choose_dataset
import seaborn as sns
from sklearn.metrics.cluster import adjusted_mutual_info_score

plt.style.use('ggplot')
plt.style.use('default')
_path = Path('./Datasets_CopMixM/')

'''

Choose the datasets.
synt_0, synt_1, synt_2, synt_3, synt_4, synt_5, synt_6, synt_3D, ais, data_breast, protein, generate.

'''
dataset_name='ais'

X , gt= choose_dataset(dataset_name,_path)

#plot of the initial dataset:
if X.shape[1]==2:
    plt.figure(figsize=[12, 9])
    plt.rcParams['xtick.labelsize'] = 20  # Dimensione del font per le etichette sull'asse x
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 30
    plt.rcParams['axes.titlesize'] = 35  # Imposta la dimensione del carattere per il titolo
      
    plt.scatter(X[:, 0], X[:, 1], c=gt, s=25, cmap='viridis', zorder=1)
    #plt.title('CopMixM BSQI')
    plt.grid(True)
    plt.show()

    
# plt.scatter(X[:,0],X[:,1],s=3)#, c=gt)
# if X.shape[1]==3:
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(X[:,0],X[:,1],X[:,2],s=2)#, c=gt)
# else:
#     plt.scatter(X[:,0],X[:,1], c=gt ,s=5, cmap='viridis', zorder=1)
#     plt.title('gt')
#     plt.show()
    
#fixed parameter
n_iter=150
tol=1e-4
parallel_iter=1
'''
if consider all copula write cop_fam=['all']
if cop_fam is not all and you want one ore 2 or three or four copula then
pass cop_fam as list i.e. cop_fam=['Gaussian'], or cop_fam=['Gaussian', 'Clayton']
'''

'''
Initialize cluster option: 'k_means','random'

for maximization the option is: 'EMP_spline', 'EMP'
'''
#choose the number of cluster and the initialization step
n_cluster=2
init_clust='random_parall'


#cop_type_opt=['all']# or ['Gaussian'] or ['Frank'] or ['Clayton'] or ['Gumbel']
cop_type_opt=['all']

'''
Maximization_type
EMP_spline
ECM
'''
maximization_type='EMP_spline'

#choose the option for the kde density estimation
opt_max_EMP_KDEpy={'kernel':'gaussian', 'bw':'ISJ'}



if init_clust!='random_parall':
    copMM=CopMixM(n_cluster,n_iter, tol, init_clust,cop_type_opt, 
                  maximization_type, opt_max_EMP_KDEpy).fit(X)
    cop_labels, clusters=copMM.predict((X))
    
    
    gmm = GaussianMixture(n_cluster,tol=1e-4, max_iter=500, covariance_type='full',init_params='random').fit(X)
    gmm=gmm.predict(X).reshape(-1,1)
    
    '''
    aa=gmm.predict_proba((X))
    cc=np.argmax(aa,1)
    bb=np.where(np.abs(aa[:,0]-aa[:,1])<5e-1)[0]
    for i in bb:
        cc[i]=random.choice([0,1])
    plt.scatter(X[:, 0], X[:, 1],c=cc ,s=5, cmap='viridis', zorder=1)
    plt.title('GMM corrected')
    plt.show()    
    '''
 
    plt.figure(figsize=[12, 9])
  
  
    plt.rcParams['xtick.labelsize'] = 20  # Dimensione del font per le etichette sull'asse x
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 30
    plt.rcParams['axes.titlesize'] = 35  # Imposta la dimensione del carattere per il titolo
   
    plt.scatter(X[:, 0], X[:, 1], c=cop_labels, s=25, cmap='viridis', zorder=1)
    #plt.title('CopMixM BSQI')
    plt.grid(True)

    
    if dataset_name=='overlap':
        #pred=np.argmax(resp, axis=1)
        #cop_labels=pred
        plt.figure(figsize=[12, 9])
        
        plt.grid(True)
        
        plt.rcParams['xtick.labelsize'] = 30  # Dimensione del font per le etichette sull'asse x
        plt.rcParams['ytick.labelsize'] = 30
        plt.rcParams['legend.fontsize'] = 30
        plt.rcParams['axes.titlesize'] = 35  # Imposta la dimensione del carattere per il titolo
        
        #plt.scatter(X[:, 0], X[:, 1], c=pred, s=5, cmap='viridis', zorder=1)
        plt.scatter(X[:, 0], X[:, 1], c=gt, s=25, cmap='viridis', zorder=1)
        plt.title('GT')

        plt.show()
    
    
    #plt.scatter(pobs(X[:, 0]), pobs(X[:, 1]),c=cop_labels ,s=3, cmap='viridis', zorder=1)
    #plt.show()
    plt.figure(figsize=[12, 9])
    plt.grid(True)
    plt.scatter(X[:, 0], X[:, 1], c=gmm ,s=25, cmap='viridis', zorder=1)
    #plt.title('GMM')
    
    plt.show()
    
    if maximization_type=='EMP':

        plt.figure(figsize=[12, 9])
        plt.grid(True)
        plt.scatter(X[:, 0], X[:, 1],c=cop_labels ,s=25, cmap='viridis', zorder=1)
        plt.title('CopMixM_KDEpy')
        plt.show()


    

    if gt is not None:#and X.shape[1]==2:
        my_accuracy = accuracy_score(gt, cop_labels, normalize=False)/ float(cop_labels.size)
        from sklearn.metrics import accuracy_score
        #accuracy_score(gt, cop_labels)
        #accuracy_score(gt, aa)
        print('Misclassification_rate CopMixM:\n',my_accuracy)
        my_accuracy = accuracy_score(gt, gmm, normalize=False) / float(cop_labels.size)
        print('Misclassification_rate Gmm:\n',my_accuracy)
    
    
    #Metrics for comparison
    
    Sil_cop = metrics.silhouette_score(X, cop_labels)
    CH_cop = metrics.calinski_harabasz_score(X, cop_labels)
    DB_cop = metrics.davies_bouldin_score(X, cop_labels)
    print('Results\n'+'Sil_CopMixM:',Sil_cop,'\nCH_CopMixM:',CH_cop,'\nDB_CopMixM:',DB_cop)
    if gt is not None:
        gt=gt.ravel()
        Rand_Score_cop=metrics.rand_score(gt, cop_labels)
        Adj_Rand_Score_cop=metrics.adjusted_rand_score(gt, cop_labels)
        homo_cop=metrics.homogeneity_score(gt, cop_labels)
        compl_cop=metrics.completeness_score(gt, cop_labels)
    
    print('\nRand_score_CopMixM',Rand_Score_cop,'\nAdj_Rand_Score_CopMixM',Adj_Rand_Score_cop, 
          '\nhomo_CopMixM', homo_cop, '\ncompleteness_CopMixM',compl_cop)
    
    
    Sil = metrics.silhouette_score(X, gmm.ravel())
    CH = metrics.calinski_harabasz_score(X, gmm.ravel())
    DB = metrics.davies_bouldin_score(X, gmm.ravel())
    
    print('Results\n'+'Sil_GMM:',Sil,'\nCH_GMM:',CH,'\nDB_GMM:',DB)
    if gt is not None:
        Rand_Score=metrics.rand_score(gt,gmm.ravel())
        Adj_Rand_Score=metrics.adjusted_rand_score(gt, gmm.ravel())
        homo=metrics.homogeneity_score(gt, gmm.ravel())
        compl=metrics.completeness_score(gt, gmm.ravel())
    print('\nRand_score_GMM',Rand_Score,'\nAdj_Rand_Score_GMM',Adj_Rand_Score, 
          '\nhomo_GMM', homo, '\ncompleteness_GMM',compl)



    from tabulate import tabulate

    # Dati per la tabella
    # data = [
    #     ["", "Silhouette Score", "Calinski-Harabasz Score", "Davies-Bouldin Score", "Adjusted Rand Score", "Homogeneity Score", "Rand Score", "Completeness Score"],
    #     ["GMM", "{:.2e}".format(Sil), "{:.2e}".format(CH), 
    #      "{:.2e}".format(DB_cop),"{:.2e}".format(Rand_Score)
    #      ,"{:.2e}".format(Adj_Rand_Score),"{:.2e}".format(homo),
    #      "{:.2e}".format(compl)],
    #     ["CopMixM_BS", "{:.2e}".format(Sil_cop), "{:.2e}".format(CH_cop), 
    #      "{:.2e}".format(DB_cop),"{:.2e}".format(Rand_Score_cop)
    #      ,"{:.2e}".format(Adj_Rand_Score_cop),"{:.2e}".format(homo_cop),
    #      "{:.2e}".format(compl_cop)],
    #     #["CopMixM_Emp", "{:.2e}".format(Sil_cop), "{:.2e}".format(CH_cop), 
    #      #"{:.2e}".format(DB_cop),"{:.2e}".format(Rand_Score_cop)
    #      #,"{:.2e}".format(Adj_Rand_Score_cop),"{:.2e}".format(homo_cop),
    #      #"{:.2e}".format(compl_cop)],
    # ]
    data = [
    ["", "GMM", "CopMixM_BS", "CopMixM_Emp"],
    ["Silhouette Score", "{:.2e}".format(Sil), "{:.2e}".format(Sil_cop), ""],
    ["Calinski-Harabasz Score", "{:.2e}".format(CH), "{:.2e}".format(CH_cop), ""],
    ["Davies-Bouldin Score", "{:.2e}".format(DB), "{:.2e}".format(DB_cop), ""],
    ['\hline'],
    ["Adjusted Rand Score", "{:.2e}".format(Adj_Rand_Score), "{:.2e}".format(Adj_Rand_Score_cop), ""],
    ["Homogeneity Score", "{:.2e}".format(homo), "{:.2e}".format(homo_cop), ""],
    ["Rand Score", "{:.2e}".format(Rand_Score), "{:.2e}".format(Rand_Score_cop), ""],
    ["Completeness Score", "{:.2e}".format(compl), "{:.2e}".format(compl_cop), ""],
]
    # Generate LaTeX table
    latex_table = tabulate(data, headers='firstrow', tablefmt='latex_raw')

    # Print LaTeX table 
    print(latex_table)
    
    #3d plot
    '''
    if dataset_name=='synt_3D':
        df = pd.DataFrame(X)
        df.columns=['X_1', 'X_2', 'X_3']
        df1=df
        df['cop_mix']=cop_labels+1
        sns.set(font_scale=1.5)
        new_title = 'Clusters'
        g=sns.pairplot(df, hue='cop_mix', palette=[ 'b','orange'], diag_kind='kde')
        g._legend.set_title(new_title)
        g.map_lower(sns.kdeplot, levels=4)
        df1=df1.drop(columns='cop_mix')
        df1['gmm']=gmm+1
        g=sns.pairplot(df1, hue='gmm', palette=[ 'b','orange'], diag_kind='kde')
        g._legend.set_title(new_title)
        g.map_lower(sns.kdeplot, levels=4)
    '''
        
else:
    copMM_n=np.zeros( (X.shape[0],parallel_iter))
    
    for ns in range(parallel_iter):   
       copMM=CopMixM(n_cluster, n_iter, tol, init_clust, cop_type_opt,
                       maximization_type,opt_max_EMP_KDEpy).fit(X)
       copMM_n[:,ns], clusters=copMM.predict(X)
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
    plt.figure(figsize=[12, 9])
  
  
    plt.rcParams['xtick.labelsize'] = 20  # Dimensione del font per le etichette sull'asse x
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 30
    plt.rcParams['axes.titlesize'] = 35  # Imposta la dimensione del carattere per il titolo
   
    plt.scatter(X[:, 0], X[:, 1], c=copMM_n, s=25, cmap='viridis', zorder=1)
    #plt.title('CopMixM BSQI')
    plt.grid(True)

    
    if dataset_name=='overlap':
        #pred=np.argmax(resp, axis=1)
        #cop_labels=pred
        plt.figure(figsize=[12, 9])
        
        plt.grid(True)
        
        plt.rcParams['xtick.labelsize'] = 30  # Dimensione del font per le etichette sull'asse x
        plt.rcParams['ytick.labelsize'] = 30
        plt.rcParams['legend.fontsize'] = 30
        plt.rcParams['axes.titlesize'] = 35  # Imposta la dimensione del carattere per il titolo
        
        #plt.scatter(X[:, 0], X[:, 1], c=pred, s=5, cmap='viridis', zorder=1)
        plt.scatter(X[:, 0], X[:, 1], c=gt, s=25, cmap='viridis', zorder=1)
        plt.title('GT')

        plt.show()
    
    
    #plt.scatter(pobs(X[:, 0]), pobs(X[:, 1]),c=cop_labels ,s=3, cmap='viridis', zorder=1)
    #plt.show()
    plt.figure(figsize=[12, 9])
    plt.grid(True)
    plt.scatter(X[:, 0], X[:, 1], c=gmm ,s=25, cmap='viridis', zorder=1)
    #plt.title('GMM')
    
    plt.show()
    
    if maximization_type=='EMP':

        plt.figure(figsize=[12, 9])
        plt.grid(True)
        plt.scatter(X[:, 0], X[:, 1],c=copMM_n ,s=25, cmap='viridis', zorder=1)
        plt.title('CopMixM_KDEpy')
        plt.show()


    

    if gt is not None:#and X.shape[1]==2:
        my_accuracy = accuracy_score(gt, copMM_n, normalize=False)/ float(copMM_n.size)
        from sklearn.metrics import accuracy_score
        #accuracy_score(gt, cop_labels)
        #accuracy_score(gt, aa)
        print('Misclassification_rate CopMixM:\n',my_accuracy)
        my_accuracy = accuracy_score(gt, gmm, normalize=False) / float(copMM_n.size)
        print('Misclassification_rate Gmm:\n',my_accuracy)
    
    
    #Metrics for comparison
    copMM_n=copMM_n.reshape(-1,)
    
    Sil_cop = metrics.silhouette_score(X, copMM_n)
    CH_cop = metrics.calinski_harabasz_score(X, copMM_n)
    DB_cop = metrics.davies_bouldin_score(X, copMM_n)
    print('Results\n'+'Sil_CopMixM:',Sil_cop,'\nCH_CopMixM:',CH_cop,'\nDB_CopMixM:',DB_cop)
    if gt is not None:
        gt=gt.ravel()
        Rand_Score_cop=metrics.rand_score(gt, copMM_n)
        Adj_Rand_Score_cop=metrics.adjusted_rand_score(gt, copMM_n)
        homo_cop=metrics.homogeneity_score(gt, copMM_n)
        compl_cop=metrics.completeness_score(gt, copMM_n)
    
    print('\nRand_score_CopMixM',Rand_Score_cop,'\nAdj_Rand_Score_CopMixM',Adj_Rand_Score_cop, 
          '\nhomo_CopMixM', homo_cop, '\ncompleteness_CopMixM',compl_cop)
    
    
    Sil = metrics.silhouette_score(X, gmm.ravel())
    CH = metrics.calinski_harabasz_score(X, gmm.ravel())
    DB = metrics.davies_bouldin_score(X, gmm.ravel())
    
    print('Results\n'+'Sil_GMM:',Sil,'\nCH_GMM:',CH,'\nDB_GMM:',DB)
    if gt is not None:
        Rand_Score=metrics.rand_score(gt,gmm.ravel())
        Adj_Rand_Score=metrics.adjusted_rand_score(gt, gmm.ravel())
        homo=metrics.homogeneity_score(gt, gmm.ravel())
        compl=metrics.completeness_score(gt, gmm.ravel())
    print('\nRand_score_GMM',Rand_Score,'\nAdj_Rand_Score_GMM',Adj_Rand_Score, 
          '\nhomo_GMM', homo, '\ncompleteness_GMM',compl)



    from tabulate import tabulate

    # Dati per la tabella
    # data = [
    #     ["", "Silhouette Score", "Calinski-Harabasz Score", "Davies-Bouldin Score", "Adjusted Rand Score", "Homogeneity Score", "Rand Score", "Completeness Score"],
    #     ["GMM", "{:.2e}".format(Sil), "{:.2e}".format(CH), 
    #      "{:.2e}".format(DB_cop),"{:.2e}".format(Rand_Score)
    #      ,"{:.2e}".format(Adj_Rand_Score),"{:.2e}".format(homo),
    #      "{:.2e}".format(compl)],
    #     ["CopMixM_BS", "{:.2e}".format(Sil_cop), "{:.2e}".format(CH_cop), 
    #      "{:.2e}".format(DB_cop),"{:.2e}".format(Rand_Score_cop)
    #      ,"{:.2e}".format(Adj_Rand_Score_cop),"{:.2e}".format(homo_cop),
    #      "{:.2e}".format(compl_cop)],
    #     #["CopMixM_Emp", "{:.2e}".format(Sil_cop), "{:.2e}".format(CH_cop), 
    #      #"{:.2e}".format(DB_cop),"{:.2e}".format(Rand_Score_cop)
    #      #,"{:.2e}".format(Adj_Rand_Score_cop),"{:.2e}".format(homo_cop),
    #      #"{:.2e}".format(compl_cop)],
    # ]
    data = [
    ["", "GMM", "CopMixM_BS", "CopMixM_Emp"],
    ["Silhouette Score", "{:.2e}".format(Sil), "{:.2e}".format(Sil_cop), ""],
    ["Calinski-Harabasz Score", "{:.2e}".format(CH), "{:.2e}".format(CH_cop), ""],
    ["Davies-Bouldin Score", "{:.2e}".format(DB), "{:.2e}".format(DB_cop), ""],
    ['\hline'],
    ["Adjusted Rand Score", "{:.2e}".format(Adj_Rand_Score), "{:.2e}".format(Adj_Rand_Score_cop), ""],
    ["Homogeneity Score", "{:.2e}".format(homo), "{:.2e}".format(homo_cop), ""],
    ["Rand Score", "{:.2e}".format(Rand_Score), "{:.2e}".format(Rand_Score_cop), ""],
    ["Completeness Score", "{:.2e}".format(compl), "{:.2e}".format(compl_cop), ""],
]
    # Generate LaTeX table
    latex_table = tabulate(data, headers='firstrow', tablefmt='latex_raw')

    # Print LaTeX table 
    print(latex_table)
    
    #3d plot
    '''
    if dataset_name=='synt_3D':
        df = pd.DataFrame(X)
        df.columns=['X_1', 'X_2', 'X_3']
        df1=df
        df['cop_mix']=cop_labels+1
        sns.set(font_scale=1.5)
        new_title = 'Clusters'
        g=sns.pairplot(df, hue='cop_mix', palette=[ 'b','orange'], diag_kind='kde')
        g._legend.set_title(new_title)
        g.map_lower(sns.kdeplot, levels=4)
        df1=df1.drop(columns='cop_mix')
        df1['gmm']=gmm+1
        g=sns.pairplot(df1, hue='gmm', palette=[ 'b','orange'], diag_kind='kde')
        g._legend.set_title(new_title)
        g.map_lower(sns.kdeplot, levels=4)
    '''
  


