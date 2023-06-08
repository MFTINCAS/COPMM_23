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


emp=1

if emp==1:
    #np.random.seed(20000)
    mu=5
    sigma=0.3
    #sample = np.random.normal(mu, sigma, 1000)
    sample=expon.rvs(size=100)
    #gamma_a=30
    #sample=gamma.rvs(a=gamma_a,size=1000)
    wd=np.ones(len(sample))
    opt_max_EMP={'emp_distr_linear':False,
                        'emp_distr_allbs':True, 
                        'distr_emp_bs2_bw':np.nan,
                        'distemp_bs2_linear':False
                        }
    epdf,ecdf,warn_ne=distbs2(sample, wd, opt_max_EMP)
    #plt.plot(sample,ecdf,'.')
    plt.plot(sample,epdf,'.')
    
    plt.hist(sample, density=True, bins=20)
    
    
    #inversa della ecdf per creare sample
    Finv=interp1d(ecdf,sample,kind='cubic',fill_value="extrapolate")
    N = 100
    u = np.random.rand(N)
    
    # Now, we need to calculate Finv(u), which will
    # gives us the samples we want.
    x = Finv(u)
    
    
    print ("Mean of samples : ", np.mean(x))
    print ("Variance of samples :", np.var(x))
    
    # Let's make a histogram of what we got. 
    
    plt.figure()
    plt.hist(x, density=True, bins=20)
    #sample = np.random.normal(mu, sigma, 1000)
    # We need to compare with samples from the Standard Normal distribution. 
    
    plt.hist(sample, density=True, bins=20, color='red',alpha=0.6)
    
    plt.legend(['Samples with our method','"Real" samples'],loc=0)
    
    #plt.show()
    test_1=ks_2samp(x, sample)
    print(test_1)
    test_2=cramervonmises_2samp(x, sample, method='auto')
    print(test_2)
    test_3=energy_distance(norm.cdf(sample, mu, sigma), ecdf, u_weights=None, v_weights=None)
    print(test_3)
    #QQplot
    plt.figure()
    plt.plot([min(sample),max(sample)],[min(sample),max(sample)],color="red")
    plt.scatter(np.sort(x), np.sort(sample))
    #plot of two cumulative
    
    #cdf=norm.cdf(sample,mu,sigma)
    cdf=expon.cdf(sample)
    #cdf=gamma.cdf(x=sample, a=gamma_a)
    plt.figure()
    plt.plot(sample, cdf,'.')
    plt.plot(sample, ecdf,'.', color='orange')
elif emp==2:
    mu=5
    sigma=0.3
    #sample = np.random.normal(mu, sigma, 1000)
    #sample=expon.rvs(size=1000)
    gamma_a=30
    sample=gamma.rvs(a=gamma_a,size=1000)
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
    test_2=cramervonmises_2samp(x, sample, method='auto')
    plt.figure()
    plt.hist(x, density=True, bins=20)
    plt.hist(sample, density=True, bins=20, color='red',alpha=0.6)
    plt.legend(['Samples with our method','"Real" samples'],loc=0)
    
    plt.figure()
    plt.plot([min(sample),max(sample)],[min(sample),max(sample)],color="red")
    plt.scatter(np.sort(x), np.sort(sample))
    #cdf=norm.cdf(sample,mu,sigma)
    #cdf=expon.cdf(sample)
    cdf=gamma.cdf(x=sample, a=gamma_a)
    
    plt.figure()
    plt.plot(sample, cdf,'.')
    plt.plot(sample, ecdf(sample),'.', color='orange')

    
    
    


