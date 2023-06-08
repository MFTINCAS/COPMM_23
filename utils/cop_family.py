# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:36:59 2021

@author: Cristiano
"""
from copulae.elliptical import GaussianCopula, StudentCopula
from copulae.archimedean import FrankCopula, ClaytonCopula, GumbelCopula

def cop_family(cop_fam_opt,dim):
    cop_fam = [
              ('Gaussian', GaussianCopula(dim)), 
              ('Student', StudentCopula(dim)),
              ('Frank', FrankCopula(dim)),
              ('Gumbel', GumbelCopula(dim)), 
              ('Clayton', ClaytonCopula(dim)),
              ]
    if cop_fam_opt=='all':
        cop_fam = cop_fam
    elif 1<len(cop_fam_opt)<5:
        a=cop_fam_opt
        cop_fam_new=[]
        for k in range(len(a)):
            for name, cop in cop_fam:
                if name==a[k]:
                    cop_fam_new.append((name,cop))
        cop_fam=cop_fam_new
    elif len(cop_fam_opt)==1:
        for name, cop in cop_fam:
            if name==cop_fam_opt[0]:
                cop_fam=(name,cop)
        
        
    return cop_fam


def cop_family_new(cop_fam_opt: list):
    cop_fam = [
              ('Gaussian', GaussianCopula), 
              #('Student', StudentCopula),
              ('Frank', FrankCopula),
              ('Gumbel', GumbelCopula), 
              ('Clayton', ClaytonCopula),
              ]
    if cop_fam_opt[0]=='all':
        cop_fam = cop_fam
    else:
        for cop in cop_fam_opt:
            a=cop_fam_opt
            cop_fam_new=[]
            for k in range(len(a)):
                for name, cop in cop_fam:
                    if name==a[k]:
                        cop_fam_new.append((name,cop))
            cop_fam=cop_fam_new      
        
    return cop_fam