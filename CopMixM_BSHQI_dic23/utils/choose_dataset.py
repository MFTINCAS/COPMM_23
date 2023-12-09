# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:27:20 2023

@author: Cristiano
"""

from pathlib import Path
import numpy as np
from data_utils.generate_dataset import generate_dataset
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler

def choose_dataset(dataset, path):
    
    if dataset=='synt_0':
        XX = pd.read_csv(path/'synt_0.csv', header=None)
        XX=XX.values
        X=XX[:,:2]
        gt=XX[:,2]
    elif dataset=='synt_1':
        XX = pd.read_csv(path/'synt_1.csv', header=None)
        XX=XX.values
        X=XX[:,:2]
        gt=XX[:,2]
    elif dataset=='synt_2':
        XX = pd.read_csv(path/'synt_2.csv', header=None)
        XX=XX.values
        X=XX[:,:2]
        gt=XX[:,2]
    elif dataset=='synt_3':
        XX = pd.read_csv(path/'synt_3.csv', header=None)
        XX=XX.values
        X=XX[:,:2]
        gt=XX[:,2]
    elif dataset=='synt_4':
        XX = pd.read_csv(path/'synt_4.csv', header=None)
        XX=XX.values
        X=XX[:,:2]
        gt=XX[:,2]
    elif dataset=='synt_5':
        XX = pd.read_csv(path/'synt_5.csv', header=None)
        XX=XX.values
        X=XX[:,:2]
        gt=XX[:,2]
    elif dataset=='synt_6':
        XX = pd.read_csv(path/'synt_6.csv', header=None)
        XX=XX.values
        X=XX[:,:2]
        gt=XX[:,2]
        
    elif dataset=='ais':
        XX = pd.read_csv(path/'ais.csv', header=None)
        XX.columns=XX.iloc[0,:]
        XX=XX.drop([0])
        X=XX.loc[:,['LBM','Wt','BMI', 'WBC','PBF']]
        X=X.values
        X=X.astype(float)
        scale = StandardScaler()
        X = scale.fit_transform(X)
        lb = LabelEncoder()
        XX['Sex'] = lb.fit_transform(XX['Sex'])
        gt=XX['Sex'].values
    elif dataset=='data_breast':
        XX = pd.read_csv(path/'data_breast.csv', header=None)
        XX.columns=XX.iloc[0,:]
        XX=XX.drop([0])
        X=XX.loc[:,['smoothness_mean','concavity_mean','concave points_mean', 'perimeter_se']]
        X=X.values
        X=X.astype(float)
        scale = StandardScaler()
        X = scale.fit_transform(X)
        lb = LabelEncoder()
        XX['diagnosis'] = lb.fit_transform(XX['diagnosis'])
        gt=XX['diagnosis'].values
    elif dataset=='overlap':
        X = pd.read_csv(path/'overlap.csv', header=None)
        XX=X.values
        X=XX[:,:2]
        gt=XX[:,2]-1
    elif dataset=='synt_3D':
        X = pd.read_csv(path/'synt_3D.csv', header=None)
        XX=X.values
        gt=XX[:,3]-1
        X=XX[:,:3]
    elif dataset=='generate':
        X, gt=generate_dataset()

    return X, gt
        
    