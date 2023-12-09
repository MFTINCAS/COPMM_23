# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 08:34:04 2021

@author: Cristiano
"""
#from emp_distribution import emp_distr
from utils.distr_emp_bs_23_new import BSemp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splrep, splder, splantider, splev, make_interp_spline
import random
from scipy.stats import norm
from scipy.stats import ks_2samp, cramervonmises_2samp, energy_distance, entropy, probplot
import statsmodels.api as sm
from scipy.stats import expon, gamma
from KDEpy import FFTKDE
from statsmodels.distributions.empirical_distribution import ECDF
from KDEpy import TreeKDE
import time
from bokeh.plotting import figure, show
import random as rm
from sklearn.metrics import mean_squared_error
import math
import scipy
import matplotlib.patches as mpatches
import pandas as pd


emp=2

n=2**15
N=n
MSE_TOT = 0
AMISE_TOT = 0
MSEqi_TOT = 0
AMISEqi_TOT = 0
ks_statistic_TOT=0
ks_statqi_TOT=0
ks_pvalue_TOT=0
ks_pvalueqi_TOT=0
CV_statistic_TOT=0
CV_statqi_TOT=0
CV_pvalue_TOT=0
CV_pvalueqi_TOT=0
time_0_TOT=[]
time_1_TOT=[]
NIT = 20
printfig = True
meth_bins = 'rice' #'scott','rice', 'fd', 'knuth_bin' 'isj' or a int
#chose the number of bins if bins is high the empirical should be overfitt
if meth_bins == 'int':
    Ne=30

for im in range(NIT):
    
        if emp == 1:
            mu=5
            sigma=0.3
            sample = np.random.normal(mu, sigma, n)
            sample=np.sort(sample)
            pdf=norm.pdf(sample, mu,sigma)
            cdf=norm.cdf(sample, mu,sigma)
            leftbw = 0
            rigthbw=0
        elif emp==2:
            loc = 0
            scale = 1
            sample=expon.rvs(size=n,loc=loc,scale=scale)
            pdf=expon.pdf(sample,loc=loc,scale=scale)
            cdf=expon.cdf(sample,loc=loc,scale=scale)
            leftbw = 1
            rigthbw=0
        else:
            leftbw = 0
            rigthbw=0
            num_gaussians = 3
            means_gaussians = [1.0, 4.0, 8.0]
            standard_deviations_gaussians = [0.5, 0.3, 0.8]
            weights_gaussians = [0.5, 0.2, 0.3]            
            # Total number of samples to generate
            num_samples = n
            samples = np.zeros(num_samples)
            for _ in range(num_samples):
                chosen_component = np.random.choice(num_gaussians, p=weights_gaussians)
                mean = means_gaussians[chosen_component]
                standard_deviation = standard_deviations_gaussians[chosen_component]
                sample = np.random.normal(loc=mean, scale=standard_deviation)
                samples[_] = sample                
            sample=np.sort(samples)
        
      
            
        sample=np.sort(sample)
      
        
        Nd = len(sample)
        
        Xdm = min(sample)
        XdM = max(sample)
        
      
       # Ne =max(3,len(np.histogram_bin_edges(sample, bins='fd')))
      
       
        if meth_bins == 'scott':
          hb=(3.49*np.std(sample))/(len(sample)**(1/3))
          Ne=int(np.floor((XdM-Xdm)/hb))+1
        elif meth_bins == 'rice':
           Ne = 1 + 2*int(Nd**(1/3))
           hb = (XdM-Xdm)/Ne
        elif meth_bins == 'knuth_bin':
           from astropy.stats import knuth_bin_width
           hb, bin_edges = knuth_bin_width(sample, return_bins=True)
           Ne=int(np.floor((XdM-Xdm)/hb))+1
        elif meth_bins == 'fd':
           Ne =max(3,len(np.histogram_bin_edges(sample, bins='fd')))
           hb = (XdM-Xdm)/Ne
        elif meth_bins == 'isj':
           kde = FFTKDE(kernel='gaussian', bw='ISJ').fit(sample)
           hb=kde.bw
           Ne =int((np.abs(Xdm-XdM))/hb)

            
         
        bins=hb
    
        sxeval=np.linspace(Xdm,XdM,num=int(N/2))
        
        if emp == 1:
          pdf=norm.pdf(sxeval, mu,sigma)
          cdf=norm.cdf(sxeval, mu,sigma)
        elif emp==2:    
          pdf=expon.pdf(sxeval,loc=loc,scale=scale)
          cdf=expon.cdf(sxeval,loc=loc,scale=scale)
        else:
           pdf = 0.0
           cdf = 0.0
           for i in range(num_gaussians):
               pdf += weights_gaussians[i] * norm.pdf(sxeval, loc=means_gaussians[i], scale=standard_deviations_gaussians[i])
           for i in range(num_gaussians):
               cdf += weights_gaussians[i] * norm.cdf(sxeval, loc=means_gaussians[i], scale=standard_deviations_gaussians[i])
      
        
        # Comparison by using the classical empirical and kdepy
        
        st1 = time.time()
        kde = FFTKDE(kernel='gaussian', bw=bins).fit(sample)
        bw=kde.bw
        
        xkde, ykde=kde.evaluate(sxeval.size)
        xkde0 = xkde.copy()
        ykde0 = ykde.copy()
        ykde = ykde[np.where( xkde <= XdM)]
        xkde = xkde[np.where( xkde <= XdM)]
        ykde = ykde[np.where( xkde >= Xdm)]
        xkde = xkde[np.where( xkde >= Xdm)]
       
        kdex=make_interp_spline(xkde,ykde,k=2)
        kde_pdf=splev(sxeval,kdex)
        kde_cdf=scipy.interpolate.splantider(kdex)
        ecdf_emp = kde_cdf(xkde)
        
        
        
        
      
        sxeval = xkde
        
        st = time.time()
        Xdm = min(sample)
        XdM = max(sample)
        
        
        bsh_data,epdf,ecdf,warn_ne= BSemp(sample, bw=Ne, sxeval=sxeval,leftbw=leftbw,rigthbw=rigthbw)
        et = time.time()
        # get the execution time
        elapsed_time0 = et - st
        print('elapsed distbs2', elapsed_time0)
        #plt.plot(sample,ecdf,'.')
   
        
        #inversa della ecdf per creare sample
        Finv=interp1d(ecdf,sxeval,kind='linear',fill_value="extrapolate")
        
        u = np.random.rand(N)
          
        # Now, we need to calculate Finv(u), which will
        # gives us the samples we want.
        x = Finv(u)
        
        
        
        print ("Mean of samples : ", np.mean(x))
        print ("Variance of samples :", np.var(x))
        
        # Let's make a histogram of what we got. 
        
        
        if emp == 1:
          pdf_k=norm.pdf(xkde, mu,sigma)
          cdf_k=norm.cdf(xkde, mu,sigma)
        elif emp==2:    
          pdf_k=expon.pdf(xkde,loc=loc,scale=scale)
          cdf_k=expon.cdf(xkde,loc=loc,scale=scale)
        else:
           pdf_k= 0.0
           cdf_k = 0.0
           for i in range(num_gaussians):
               pdf_k += weights_gaussians[i] * norm.pdf(xkde, loc=means_gaussians[i], scale=standard_deviations_gaussians[i])
           for i in range(num_gaussians):
               cdf_k += weights_gaussians[i] * norm.cdf(xkde, loc=means_gaussians[i], scale=standard_deviations_gaussians[i])
      
        
        
        #ecdf_emp=ECDF(xkde)
        et1 = time.time()
        # get the execution time
        elapsed_time1 = et1 - st1
        
        Finv=interp1d(ecdf_emp,sxeval,fill_value="extrapolate")
       
        u = np.random.rand(N)
        x_emp = Finv(u)
        
        if printfig and im==NIT-1:
            plt.figure(figsize=[10,8])
            
            plt.grid(True)
            
  
            plt.rcParams['xtick.labelsize'] = 15  # Dimensione del font per le etichette sull'asse x
            plt.rcParams['ytick.labelsize'] = 15
            plt.rcParams['legend.fontsize'] = 15
            plt.fill_between(sxeval, pdf_k, fc='silver', label='Real Dist',alpha=0.9)
            #silv_patch = mpatches.Patch(color='silver', label='Samples Real')
            plt.hist(sample, density=True, alpha=0.3, bins=20, rwidth=2,ec="black",fc="yellow",label='Histogram')
            #hist_patch = mpatches.Patch(color='yellow', alpha=0.4,label='Histogram')
           # plt.plot(sxeval,kde_pdf,'-g',label='Samples KDE')
            plt.plot(xkde,ykde,'-g',label='Samples KDE')
            #green_patch = mpatches.Patch(color='green', )
            plt.plot(sxeval,epdf,'-r',markersize=0.2,linewidth=1,label='Samples BSHQI')
            #red_patch = mpatches.Patch(color='red', )
            plt.plot(sample, np.full_like(sample, -0.02), '|k', markeredgewidth=1)
            plt.legend()            
            plt.figure()
            
            plt.figure(figsize=[10,8])
            plt.rcParams['xtick.labelsize'] = 15  # Dimensione del font per le etichette sull'asse x
            plt.rcParams['ytick.labelsize'] = 15
            plt.rcParams['legend.fontsize'] = 15
            plt.grid(True)
            plt.hist(x, density=True, bins=50)
            #sample = np.random.normal(mu, sigma, 1000)
            # We need to compare with samples from the Standard Normal distribution. 
            
            plt.hist(sample, density=True, bins=50, color='red',alpha=0.6)
            plt.hist(x_emp, density=True, bins=50, color='green',alpha=0.6)
            plt.legend(['Samples BSHQI','Real samples', 'Samples Empirical'],loc=0)
    
        
        #plot of  cumulatives
 
            plt.figure(figsize=[10,8])
            plt.rcParams['xtick.labelsize'] = 15  # Dimensione del font per le etichette sull'asse x
            plt.rcParams['ytick.labelsize'] = 15
            plt.rcParams['legend.fontsize'] = 15
        
            plt.grid(True)
            plt.plot(sxeval, cdf_k,'-b',label='Real CDF')
            #blue_patch = mpatches.Patch(color='blue', label='Real CDF')
            plt.plot(sxeval, ecdf,'-', color='orange',label='BSHQI ECDF')
            #red_patch = mpatches.Patch(color='orange', label='BSHQI ECDF')
            plt.plot(sxeval,ecdf_emp,'-', c='green', label='ECDF KDE')
            #green_patch = mpatches.Patch(color='green', label='ECDF KDE')
            #plt.legend(['Real cdf','BSHQI ecdf', 'ECDF'],loc=0)
            plt.legend(loc='center right')#handles=[blue_patch,red_patch,green_patch])
            
        print('QI_spline:')
        test_1_bs=ks_2samp(x, sample,mode='asymp')
        ks_statqi_TOT=ks_statqi_TOT+test_1_bs[0]
        ks_pvalueqi_TOT=ks_pvalueqi_TOT+test_1_bs[1]
        print(test_1_bs)
        test_2_bs=cramervonmises_2samp(x, sample, method='auto')
        CV_statqi_TOT=CV_statqi_TOT+test_2_bs.statistic
        CV_pvalueqi_TOT=CV_pvalueqi_TOT+test_2_bs.pvalue
        print(test_2_bs)
        #test_3=energy_distance(norm.cdf(sample, mu, sigma), ecdf, u_weights=None, v_weights=None)
    #    print(test_3)
        MSE = mean_squared_error(pdf_k, epdf)
        print("MSE QI", MSE)
        RMSE = math.sqrt(MSE)
        print("RMSE QI", RMSE)
        amise = (1 / n) * np.sum((epdf - pdf_k)**2)*hb
        MSEqi_TOT= MSEqi_TOT+MSE
        AMISEqi_TOT = AMISEqi_TOT + amise
    
        print("AMISE_QI:", amise)
        
        print('Gauusian Kernel:')
        test_1=ks_2samp(x_emp, sample,mode='asymp')
        ks_statistic_TOT=ks_statistic_TOT+test_1[0]
        ks_pvalue_TOT=ks_pvalue_TOT+test_1[1]
        print(test_1)
        test_2=cramervonmises_2samp(x_emp, sample, method='auto')
        CV_statistic_TOT=CV_statistic_TOT+test_2.statistic
        CV_pvalue_TOT=CV_pvalue_TOT+test_2.pvalue
        print(test_2)
        #test_3=energy_distance(norm.cdf(sample, mu, sigma), ecdf_emp(sample), u_weights=None, v_weights=None)
        #print(test_3)
        MSE = mean_squared_error(pdf, kde_pdf)
        MSE = mean_squared_error(pdf_k, ykde)
        print('MSE EGK:', MSE)
        RMSE = math.sqrt(MSE)
        print('RMSE EGK:', RMSE)
        amise = (1 / n) * np.sum((kde_pdf - pdf)**2)*hb
        amise = (1 / n) * np.sum((ykde - pdf_k)**2)*hb
        MSE_TOT= MSE_TOT+MSE
        AMISE_TOT = AMISE_TOT + amise
    
        print("AMISE_EGK:", amise)
        

            
        et = time.time()
        time_0_TOT.append(elapsed_time0)
        time_1_TOT.append(elapsed_time1)
    
        # get the execution time
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
        print('Execution time0:', elapsed_time0, 'seconds')
        print('Execution time1:', elapsed_time1, 'seconds')
        

        
        
MSEqi_TOT = MSEqi_TOT/NIT
AMISEqi_TOT = AMISEqi_TOT/NIT
MSE_TOT = MSE_TOT/NIT
AMISE_TOT = AMISE_TOT/NIT



ks_statqi_TOT=ks_statqi_TOT/NIT
ks_statistic_TOT=ks_statistic_TOT/NIT
CV_statqi_TOT=CV_statqi_TOT/NIT
CV_statistic_TOT=CV_statistic_TOT/NIT

ks_pvalueqi_TOT=ks_pvalueqi_TOT/NIT
ks_pvalue_TOT=ks_pvalue_TOT/NIT
CV_pvalueqi_TOT=CV_pvalueqi_TOT/NIT
CV_pvalue_TOT=CV_pvalue_TOT/NIT



time_0_TOT_mean=np.mean(np.stack(time_0_TOT))
time_1_TOT_mean=np.mean(np.stack(time_1_TOT))


time_0_TOT_std=np.std(np.stack(time_0_TOT))
time_1_TOT_std=np.std(np.stack(time_1_TOT))
    
print('Execution time0 TOT:', time_0_TOT_mean, 'seconds')
print('Execution time1 TOT:', time_1_TOT_mean, 'seconds')

print('Execution time0 TOT_std:', time_0_TOT_std, 'seconds')
print('Execution time1 TOT_std:', time_1_TOT_std, 'seconds')


print('MSE_TOT', MSE_TOT)
print('AMISE_TOT', AMISE_TOT)


print('MSEqi_TOT', MSEqi_TOT)
print('AMISEqi_TOT', AMISEqi_TOT)



from tabulate import tabulate

# Data for the table
data = [
    ["", "AMISE", "RMSE", "KS-Test", "", "Cramér–von Mises", ""],
    ["", "", "", "statistic", "p-value", "statistic", "p-value"],
    ["BSHQI", "{:.2e}".format(AMISEqi_TOT), "{:.2e}".format(MSEqi_TOT),
     "{:.2e}".format(ks_statqi_TOT), "{:.2e}".format(ks_pvalueqi_TOT),
     "{:.2e}".format(CV_statqi_TOT), "{:.2e}".format(CV_pvalueqi_TOT)],
    ["\hline"],
    ["EMPIRICAL", "{:.2e}".format(AMISE_TOT), "{:.2e}".format(MSE_TOT), "{:.2e}".format(ks_statistic_TOT),
     "{:.2e}".format(ks_pvalue_TOT),"{:.2e}".format(CV_statistic_TOT), "{:.2e}".format(CV_pvalue_TOT)]
]

# Generate LaTeX table
latex_table = tabulate(data, headers='firstrow', tablefmt='latex_raw')

# Print LaTeX table
print(latex_table)