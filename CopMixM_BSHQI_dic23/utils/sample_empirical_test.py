# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 08:34:04 2021

@author: Cristiano
"""
from utils.emp_distribution import emp_distr
from utils.distr_emp_bs import distempcont_bs2 as distbs2
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
from KDEpy import FFTKDE
import time
from bokeh.plotting import figure, show
import random as rm
from sklearn.metrics import mean_squared_error
import math
import scipy
import matplotlib.patches as mpatches
import pandas as pd


emp=1
n=100000
N=n


if emp==1:
    #np.random.seed(20000)
    mu=5
    sigma=0.3
    sample = np.random.normal(mu, sigma, n)
    sample=np.sort(sample)
 

    wd=np.ones(len(sample))

    st = time.time()
    epdf,ecdf,warn_ne=distbs2(sample, wd)
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    #plt.plot(sample,ecdf,'.')
    pdf=norm.pdf(sample, mu,sigma)
    
    #inversa della ecdf per creare sample
    Finv=interp1d(ecdf,sample,kind='linear',fill_value="extrapolate")
    
    u = np.random.rand(N)
    
    bins=(max(sample)-min(sample))*1e-1#Mazzia
    #scelta numero bins metodo scott
    # hb=int(3.5*np.std(sample))/(len(sample)**(1/3)) 
    # hb_2=int(10.5*np.std(sample))/(len(sample)**(1/3)) 
    Xdm = min(sample)
    XdM = max(sample)
    # Ne=int(np.floor((XdM-Xdm)/hb))
    # bins=hb
    
    Ne =  1 + int( (n**(1/3)))#scelta bandwidth
    bins=((XdM-Xdm)/Ne)
    hb=bins
    #metodo SIlverman
    # sigma = np.std(sample)

    # # Calcola l'intervallo interquartile (IQR)
    # q75, q25 = np.percentile(sample, [75 ,25])
    # IQR = q75 - q25

    # # Numero di osservazioni

    # # Calcola la larghezza della banda con la regola di Silverman
    # h_silverman = 0.9 * min(sigma / 1.34, IQR / (1.34 * n**(-1/5)))
    # bins=h_silverman


    
    #bins=0.1015
    
    # Now, we need to calculate Finv(u), which will
    # gives us the samples we want.
    x = Finv(u)
    
    
    print ("Mean of samples : ", np.mean(x))
    print ("Variance of samples :", np.var(x))
    
    # Let's make a histogram of what we got. 
    # Comparison by using the classical empirical and kdepy
    

    st1 = time.time()
    kde = FFTKDE(kernel='gaussian', bw=bins).fit(sample)
    xkde,ykde=kde.evaluate()
    kde=make_interp_spline(xkde,ykde,k=2)
    kde_pdf=splev(sample,kde)
    kde_cdf=scipy.interpolate.splantider(kde)
    ecdf_emp = kde_cdf(sample)
    
    #ecdf_emp=ECDF(xkde)
    et1 = time.time()
    # get the execution time
    elapsed_time1 = et1 - st1
    
    Finv=interp1d(ecdf_emp,sample,fill_value="extrapolate")
   
    u = np.random.rand(N)
    x_emp = Finv(u)
    
    plt.grid(True)
    plt.fill_between(sample, norm(mu, sigma).pdf(sample), fc='silver', label='Real Dist',alpha=0.9)
    silv_patch = mpatches.Patch(color='silver', label='Samples Real')
    plt.hist(sample, density=True, alpha=0.3, bins=20, rwidth=2,ec="black",fc="yellow")
    hist_patch = mpatches.Patch(color='yellow', label='histogram', alpha=0.4)
    plt.plot(sample,kde_pdf,'-g',markersize=0.2)
    green_patch = mpatches.Patch(color='green', label='Samples KDE')
    plt.plot(sample,epdf,'-r',markersize=0.2,linewidth=1)
    red_patch = mpatches.Patch(color='red', label='Samples BSHQI')
    plt.plot(sample, np.full_like(sample, -0.02), '|k', markeredgewidth=1)
    #plt.plot(sample,pdf,'.',markersize=0.6)
    #plt.hist(sample, density=True, bins=50, ec="blue",fc="green", alpha=0.3)
    #green_patch = mpatches.Patch(color='green', label='Real Sample', alpha=0.3)

    #x_plot = np.linspace(np.min(sample) * 0.95, np.max(sample) * 1.05, 10000)

    plt.legend(handles=[green_patch,red_patch,silv_patch, hist_patch])

    #"plt.legend(['Samples Empirical','Samples BSHQI','Real samples'],loc=0)
    
    
    
    
    plt.figure()
    plt.grid(True)
    plt.hist(x, density=True, bins=50)
    #sample = np.random.normal(mu, sigma, 1000)
    # We need to compare with samples from the Standard Normal distribution. 
    
    plt.hist(sample, density=True, bins=50, color='red',alpha=0.6)
    
    plt.hist(x_emp, density=True, bins=50, color='green',alpha=0.6)
    
    plt.legend(['Samples BSHQI','Real samples', 'Samples Empirical'],loc=0)
    
    #plt.show()
    print('QI_spline:')
    test_1_qi=ks_2samp(x, sample,method='asymp')
    print(test_1_qi)
    test_2_qi=cramervonmises_2samp(x, sample, method='auto')
    print(test_2_qi)
    #test_3=energy_distance(norm.cdf(sample, mu, sigma), ecdf, u_weights=None, v_weights=None)
#    print(test_3)
    MSE_QI = mean_squared_error(pdf, epdf)
    RMSE_QI = (MSE_QI)
    print("RMSE QI", RMSE_QI)
    amise = (1 / n) * np.sum((epdf - pdf)**2)*hb

    print("AMISE_QI:", amise)
    
    print('Empirical:')
    test_1=ks_2samp(x_emp, sample,method='auto')
    print(test_1)
    test_2=cramervonmises_2samp(x_emp, sample, method='auto')
    print(test_2)
    #test_3=energy_distance(norm.cdf(sample, mu, sigma), ecdf_emp(sample), u_weights=None, v_weights=None)
    #print(test_3)
    MSE = mean_squared_error(pdf, kde_pdf)
    RMSE = (MSE)
    print('RMSE EMP:', RMSE)
    amise = (1 / n) * np.sum((kde_pdf - pdf)**2)*hb

    print("AMISE_EMP:", amise)
    
    data = {
    'Test Type': ['QI Spline KS Test', 'QI Spline Cramer-von Mises Test', 'RMSE QI',
                  'Empirical KS Test', 'Empirical Cramer-von Mises Test', 'RMSE Empirical'],
    'Test Result': [test_1_qi.pvalue, test_2_qi.pvalue, RMSE_QI, test_1.pvalue, test_2.pvalue, RMSE]
    }

    # Crea un DataFrame
    df = pd.DataFrame(data)

    # Salva il DataFrame come un file CSV
    df.to_csv('statistiche.csv', index=False)
    #QQplot
    plt.figure()
    
    plt.plot([min(sample),max(sample)],[min(sample),max(sample)],color="red")
    plt.scatter(np.sort(x), sample)
    plt.scatter(np.sort(x_emp),sample, c='green')
    #plot of two cumulative
    
    cdf=norm.cdf(sample,mu,sigma)
    #cdf=expon.cdf(sample)
    #cdf=gamma.cdf(x=sample, a=gamma_a)
    plt.figure()
    plt.grid(True)
    plt.plot(sample, cdf,'-b',markersize=0.6)
    blue_patch = mpatches.Patch(color='blue', label='Real CDF')
    plt.plot(sample, ecdf,'-', color='orange',markersize=0.6)
    red_patch = mpatches.Patch(color='orange', label='BSHQI ECDF')
    plt.plot(sample,ecdf_emp,'-', c='green',markersize=0.6)
    green_patch = mpatches.Patch(color='green', label='ECDF KDE')
    #plt.legend(['Real cdf','BSHQI ecdf', 'ECDF'],loc=0)
    plt.legend(handles=[blue_patch,red_patch,green_patch])
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print('Execution time1:', elapsed_time1, 'seconds')
    

    
    
if emp==2:
    sample=expon.rvs(size=n)
    sample=np.sort(sample)
    wd=np.ones(len(sample))
    opt_max_EMP={'emp_distr_linear':False,
                        'emp_distr_allbs':True, 
                        'distr_emp_bs2_bw':np.nan,
                        'distemp_bs2_linear':False
                        }
    st = time.time()
    epdf,ecdf,warn_ne=distbs2(sample, wd, opt_max_EMP)
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    #plt.plot(sample,ecdf,'.')
    pdf=expon.pdf(sample)
    
    
    #inversa della ecdf per creare sample
    Finv=interp1d(ecdf,sample,kind='linear',fill_value="extrapolate")
    
    u = np.random.rand(N)
    
    bins=(max(sample)-min(sample))*1e-1#Mazzia
    
    #bins=0.1015
    
    hb=int(3.5*np.std(sample))/(len(sample)**(1/3)) 
    Xdm = min(sample)
    XdM = max(sample)
    Ne=int(np.floor((XdM-Xdm)/hb))
    bins=hb
    
    # Now, we need to calculate Finv(u), which will
    # gives us the samples we want.
    x = Finv(u)
    
    
    print ("Mean of samples : ", np.mean(x))
    print ("Variance of samples :", np.var(x))
    
    # Let's make a histogram of what we got. 
    # Comparison by using the classical empirical and kdepy
    
    st1 = time.time()
    kde = FFTKDE(kernel='gaussian', bw=bins).fit(sample)
    xkde,ykde=kde.evaluate()
    kde=make_interp_spline(xkde,ykde,k=2)
    kde_pdf=splev(sample,kde)
    kde_cdf=scipy.interpolate.splantider(kde)
    ecdf_emp = kde_cdf(sample)
    
    #ecdf_emp=ECDF(xkde)
    et1 = time.time()
    # get the execution time
    elapsed_time1 = et1 - st1
    
    Finv=interp1d(ecdf_emp,sample,fill_value="extrapolate")
   
    u = np.random.rand(N)
    x_emp = Finv(u)
    
    plt.grid(True)
    plt.fill_between(sample, expon.pdf(sample), fc='silver', label='Real Dist',alpha=0.9)
    #silv_patch = mpatches.Patch(color='silver', label='Samples Real')
    plt.hist(sample, density=True, alpha=0.3, bins=20, rwidth=2,ec="black",fc="yellow",label='Histogram')
    #hist_patch = mpatches.Patch(color='yellow', alpha=0.4,label='Histogram')
    plt.plot(sample,kde_pdf,'-g',label='Samples KDE')
    #green_patch = mpatches.Patch(color='green', )
    plt.plot(sample,epdf,'-r',markersize=0.2,linewidth=1,label='Samples BSHQI')
    #red_patch = mpatches.Patch(color='red', )
    plt.plot(sample, np.full_like(sample, -0.02), '|k', markeredgewidth=1)
    
    # plt.plot(sample,kde_pdf,'.b',markersize=0.6)
    # blue_patch = mpatches.Patch(color='blue', label='Samples KDE')
    # plt.plot(sample,epdf,'.r',markersize=0.6)
    # red_patch = mpatches.Patch(color='red', label='Samples BSHQI')
    # plt.plot(sample, np.full_like(sample, -0.02), '|k', markeredgewidth=1)
    # #plt.plot(sample,pdf,'.',markersize=0.6)
    # #plt.hist(sample, density=True, bins=50, ec="blue",fc="green", alpha=0.3)
    # #green_patch = mpatches.Patch(color='green', label='Real Sample', alpha=0.3)
    # #x_plot = np.linspace(np.min(sample) * 0.95, np.max(sample) * 1.05, 10000)
    # plt.fill_between(sample, expon.pdf(sample), fc='silver', label='Real Dist',alpha=0.7)
    # silv_patch = mpatches.Patch(color='silver', label='Samples Real')
    # plt.legend(handles=[blue_patch,red_patch,silv_patch])
    # #"plt.legend(['Samples Empirical','Samples BSHQI','Real samples'],loc=0)
    
    plt.legend()
    
    
    plt.figure()
    plt.grid(True)
    plt.hist(x, density=True, bins=50)
    #sample = np.random.normal(mu, sigma, 1000)
    # We need to compare with samples from the Standard Normal distribution. 
    
    plt.hist(sample, density=True, bins=50, color='red',alpha=0.6)
    plt.hist(x_emp, density=True, bins=50, color='green',alpha=0.6)
    plt.legend(['Samples BSHQI','Real samples', 'Samples Empirical'],loc=0)
    
    #plt.show()
    print('QI_spline:')
    test_1=ks_2samp(x, sample,method='asymp')
    print(test_1)
    test_2=cramervonmises_2samp(x, sample, method='auto')
    print(test_2)
    #test_3=energy_distance(norm.cdf(sample, mu, sigma), ecdf, u_weights=None, v_weights=None)
#    print(test_3)
    MSE = mean_squared_error(pdf, epdf)
    RMSE = math.sqrt(MSE)
    print("RMSE QI", RMSE)
    amise = (1 / n) * np.sum((epdf - pdf)**2)*hb

    print("AMISE_QI:", amise)
    
    print('Empirical:')
    test_1=ks_2samp(x_emp, sample,method='auto')
    print(test_1)
    test_2=cramervonmises_2samp(x_emp, sample, method='auto')
    print(test_2)
    #test_3=energy_distance(norm.cdf(sample, mu, sigma), ecdf_emp(sample), u_weights=None, v_weights=None)
    #print(test_3)
    MSE = mean_squared_error(pdf, kde_pdf)
    RMSE = math.sqrt(MSE)
    print('RMSE EMP:', RMSE)
    amise = (1 / n) * np.sum((kde_pdf - pdf)**2)*hb

    print("AMISE_EMP:", amise)
    
    #plot of  cumulatives
    
    cdf=expon.cdf(sample)
    plt.figure(figsize=[10,8])
    plt.rcParams['xtick.labelsize'] = 15  # Dimensione del font per le etichette sull'asse x
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 15

    plt.grid(True)
    plt.plot(sample, cdf,'-b',label='Real CDF')
    #blue_patch = mpatches.Patch(color='blue', label='Real CDF')
    plt.plot(sample, ecdf,'-', color='orange',label='BSHQI ECDF')
    #red_patch = mpatches.Patch(color='orange', label='BSHQI ECDF')
    plt.plot(sample,ecdf_emp,'-', c='green', label='ECDF KDE')
    #green_patch = mpatches.Patch(color='green', label='ECDF KDE')
    #plt.legend(['Real cdf','BSHQI ecdf', 'ECDF'],loc=0)
    plt.legend(loc='center right')#handles=[blue_patch,red_patch,green_patch])
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
    print('Execution time1:', elapsed_time1, 'seconds')
    
    
elif emp==3:#non necessaria

    gamma_a=30
    sample=gamma.rvs(a=gamma_a,size=1000)
    sample=np.sort(sample)
    kde = FFTKDE(kernel='gaussian', bw='ISJ').fit(sample)
    x,y=kde.evaluate()
    kde=make_interp_spline(x,y,k=1)
    kde_pdf=splev(sample,kde)
    plt.plot(sample,kde_pdf,'.')
    plt.hist(sample, density=True, bins=40)
    
    ecdf=ECDF(sample)
    
    Finv=interp1d(ecdf.y,ecdf.x,fill_value="extrapolate")
    N = 1000
    u = np.random.rand(N)
    x = Finv(u)
    test_1=ks_2samp(x, sample)
    print(test_1)
    test_2=cramervonmises_2samp(x, sample, method='auto')
    print(test_2)
    plt.figure()
    plt.hist(x, density=True, bins=20)
    plt.hist(sample, density=True, bins=20, color='red',alpha=0.6)
    plt.legend(['Samples with our method','"Real" samples'],loc=0)
    
    plt.figure()
    plt.plot([min(sample),max(sample)],[min(sample),max(sample)],color="red")
    plt.scatter(np.sort(x), sample)
    cdf=gamma.cdf(x=sample, a=gamma_a)
    
    plt.figure()
    plt.plot(sample, cdf,'.')
    plt.plot(sample, ecdf(sample),'.', color='orange')
    
    
    
elif emp==4:
    # Definire i parametri delle Gaussiane
    num_gaussians = 3
    means_gaussians = [1.0, 4.0, 8.0]
    standard_deviations_gaussians = [0.5, 0.3, 0.8]
    weights_gaussians = [0.5, 0.2, 0.3]
    
    # Total number of samples to generate
    num_samples = n
    
    # Generate samples from the mixture distribution
    samples = np.zeros(num_samples)
    
    for _ in range(num_samples):
        chosen_component = np.random.choice(num_gaussians, p=weights_gaussians)
        mean = means_gaussians[chosen_component]
        standard_deviation = standard_deviations_gaussians[chosen_component]
        sample = np.random.normal(loc=mean, scale=standard_deviation)
        samples[_] = sample       
        
    samples=np.sort(samples)
    
    wd=np.ones(len(samples))
    opt_max_EMP={'emp_distr_linear':False,
                        'emp_distr_allbs':True, 
                        'distr_emp_bs2_bw':np.nan,
                        'distemp_bs2_linear':False
                        }

    epdf,ecdf,warn_ne=distbs2(samples, wd)

    # Calculate the exact PDF
    #x = np.linspace(min(samples), max(samples), N)
    pdf_exact = 0.0
    for i in range(num_gaussians):
        pdf_exact += weights_gaussians[i] * norm.pdf(samples, loc=means_gaussians[i], scale=standard_deviations_gaussians[i])
        
    
    #inversa della ecdf per creare sample
    Finv=interp1d(ecdf,samples,kind='linear',fill_value="extrapolate")
    u = np.random.rand(N)
    
    #bins=(max(samples)-min(samples))*1e-1#Mazzia
    #bins=0.1015
    hb=int(3.5*np.std(samples))/(len(samples)**(1/3)) 
    Xdm = min(samples)
    XdM = max(samples)
    Ne=int(np.floor((XdM-Xdm)/hb))
    bins=hb
    
    # Now, we need to calculate Finv(u), which will
    # gives us the samples we want.
    x = Finv(u)
    
    # Let's make a histogram of what we got. 
    # Comparison by using the classical empirical and kdepy
    
    
    kde = FFTKDE(kernel='gaussian', bw=bins).fit(samples)
    xkde,ykde=kde.evaluate()
    kde=make_interp_spline(xkde,ykde,k=2)
    kde_pdf=splev(samples,kde)
    kde_cdf=scipy.interpolate.splantider(kde)
    ecdf_emp = kde_cdf(samples)
    
    #ecdf_emp=ECDF(xkde)
    
    
    
    Finv=interp1d(ecdf_emp,samples,fill_value="extrapolate")
   
    u = np.random.rand(N)
    x_emp = Finv(u)
    

    plt.grid(True)
    plt.fill_between(np.sort(samples), pdf_exact, fc='silver', label='Real Dist',alpha=0.9)
    silv_patch = mpatches.Patch(color='silver', label='Samples Real')
    plt.hist(samples, density=True, alpha=0.3, bins=20, rwidth=2,ec="black",fc="yellow")
    hist_patch = mpatches.Patch(color='yellow', label='histogram', alpha=0.4)
    plt.plot(samples,kde_pdf,'-g',markersize=0.2)
    green_patch = mpatches.Patch(color='green', label='Samples KDE')
    plt.plot(samples,epdf,'-r',markersize=0.2,linewidth=1)
    red_patch = mpatches.Patch(color='red', label='Samples BSHQI')
    plt.plot(samples, np.full_like(samples, -0.02), '|k', markeredgewidth=1)
    plt.legend(handles=[green_patch,red_patch,silv_patch, hist_patch])
    
    
    
    plt.figure()
    plt.grid(True)
    plt.hist(x, density=True, bins=50)
    # We need to compare with samples from the mixture Normal distribution. 
    plt.hist(samples, density=True, bins=50, color='red',alpha=0.6)
    plt.hist(x_emp, density=True, bins=50, color='green',alpha=0.6)
    plt.legend(['Samples BSHQI','Real samples', 'Samples Empirical'],loc=0)
    
    #plt.show()
    print('QI_spline:')
    test_1=ks_2samp(x, samples,method='asymp')
    print(test_1)
    test_2=cramervonmises_2samp(x, samples, method='auto')
    print(test_2)
    #test_3=energy_distance(norm.cdf(sample, mu, sigma), ecdf, u_weights=None, v_weights=None)
    #print(test_3)
    MSE = mean_squared_error(pdf_exact, epdf)
    RMSE = math.sqrt(MSE)
    print("RMSE QI", RMSE)
    
    print('Empirical:')
    test_1=ks_2samp(x_emp, samples,method='auto')
    print(test_1)
    test_2=cramervonmises_2samp(x_emp, samples, method='auto')
    print(test_2)
    #test_3=energy_distance(norm.cdf(sample, mu, sigma), ecdf_emp(sample), u_weights=None, v_weights=None)
    #print(test_3)
    MSE = mean_squared_error(pdf_exact, kde_pdf)
    RMSE = math.sqrt(MSE)
    print('RMSE EMP:', RMSE)
    

    
    # Calculate the cumulative distribution function (CDF)
    cdf = np.arange(1, num_samples + 1) / num_samples
    #cdf=expon.cdf(sample)
    #cdf=gamma.cdf(x=sample, a=gamma_a)
    plt.figure()
    plt.grid(True)
    plt.plot(samples, cdf,'-b',markersize=0.6)
    blue_patch = mpatches.Patch(color='blue', label='Real CDF')
    plt.plot(samples, ecdf,'-', color='orange',markersize=0.6)
    red_patch = mpatches.Patch(color='orange', label='BSHQI ECDF')
    plt.plot(samples,ecdf_emp,'-', c='green',markersize=0.6)
    green_patch = mpatches.Patch(color='green', label='ECDF KDE')
    #plt.legend(['Real cdf','BSHQI ecdf', 'ECDF'],loc=0)
    plt.legend(handles=[blue_patch,red_patch,green_patch])
    