# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 23:55:56 2021

@author: Cristiano
"""

import cv2
import numpy as np
from sklearn.datasets import make_blobs
from os.path import isfile
import pandas
from os.path import dirname, join as pjoin
from pathlib import Path
from os import listdir
from os.path import isfile, join, exists
import random
from skimage import data, img_as_float
from skimage import exposure
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, IncrementalPCA
#from pyod.models.copod import COPOD


def result_table(dataset, path_result, conf_cop, conf_gauss, conf_cop1, conf_gauss1, metrics_cop, metrics_cop1, metrics_gmm, metrics_gmm1):
    file=(path_result/"summary").with_suffix(".csv")
    with open(file, 'a') as f:
        #header="Dataset;Mode;Copula;opt_max;clean_map;OA;F1;Mar;\n"
        #f.write(header)
        name="%s;%s;%s;%s;%s;"%(dataset,conf_cop[0],conf_cop[1],conf_cop[2], conf_cop[3])
        values="%.2f;%.2f;%.2f;\n"%(metrics_cop[0][0],metrics_cop[0][1],metrics_cop[0][2])
        name1="%s;%s;%s;%s;%s;"%(dataset,conf_gauss[0], ' ' ,' ', conf_gauss[1])
        values1="%.2f;%.2f;%.2f;\n"%(metrics_gmm[0][0],metrics_gmm[0][1],metrics_gmm[0][2])
        name2="%s;%s;%s;%s;%s;"%(dataset,conf_cop1[0],conf_cop1[1],conf_cop1[2], conf_cop1[3])
        values2="%.2f;%.2f;%.2f;\n"%(metrics_cop1[0][0],metrics_cop1[0][1],metrics_cop1[0][2])
        name3="%s;%s;%s;%s;%s;"%(dataset,conf_gauss1[0], ' ' ,' ', conf_gauss1[1])
        values3="%.2f;%.2f;%.2f;\n"%(metrics_gmm1[0][0],metrics_gmm1[0][1],metrics_gmm1[0][2])
        f.write(name+values+name1+values1+name2+values2+name3+values3)

def test_dataset_auto(path):
    
    path_auto=Path(path/'dataset_appice_cluster/input_clust/')
    path_gt=Path(path/'dataset_appice_cluster/input_clust/gt')
    files = [f for f in listdir(path_auto) if isfile(join(path_auto, f))]
    files_gt = [gt for gt in listdir(path_gt) if isfile(join(path_gt, gt))]
    return files, files_gt

    
#pulizia mappa
def clean_map(change_map):
    change_map = change_map.astype(np.uint8)
    #kernel = np.asarray(((0,0,4,0,0),
    #(0,1,1,1,45),
    #(1,1,1,1,1),
    #(0,111,1,10,0),
    #(0,0,1,0,0)), dtype=np.uint8)
    #kernel = np.ones((10, 10), np.uint8)
    kernel = np.asarray(((1,1,1),
    (1,1,1),
    (1,1,1)), dtype=np.uint8)
    cleanChangeMap = cv2.erode(change_map,kernel)
    return cleanChangeMap



def generate_dataset():
    X, y=make_blobs(n_samples=1000, random_state=100, centers=3, n_features=2, center_box=(0, 20))
    theta = np.radians(60)
    t = np.tan(theta)
    shear_x = np.array(((1, t), (0, 1))).T
    X = X.dot(shear_x)
    X=np.array((X))
    return X, y

def split_random(len_dataset, w):
    len_dataset=np.arange(len_dataset)
    len_dataset=sorted(len_dataset, key=lambda len_dataset: random.random())
    a=np.array_split(len_dataset,w)
    return a

def ecdf_w(data, w):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y =((w/(sum(w)))*np.arange(1, n+1) /n)
    return(x,y)

def wecdf(data, weights):#weighted ecdf
    data = np.asarray(data)
    mask = ~np.isnan(data)
    data = data[mask]
    sort = np.argsort(data)
    data = data[sort]
    if weights is None:
        # Ensure that we end at exactly 1, avoiding floating point errors.
        cweights = (1 + np.arange(len(data))) / len(data)
    else:
        weights = weights[mask]
        weights = weights / np.sum(weights)
        weights = weights[sort]
        cweights = np.cumsum(weights)
    return data, cweights

def spatialCorrection(A, windowSize=1):
        print("------Spatial Correction new")
        row, col = A.shape
        C= np.copy(A)
        Acount1 =np.zeros(A.shape)
        Acount0 =np.zeros(A.shape) 
        windArea=np.zeros(A.shape) 
        Acount1o =np.zeros(A.shape)
        Acount0o =np.zeros(A.shape) 
        windowArea=np.zeros(A.shape) 
        for i in range(-windowSize, +windowSize+1):
            for j in range(-windowSize, +windowSize+1):
                rowS=max(0,i)
                rowE=min(row,row+i)
                colS=max(0,j)
                colE=min(col,col+j)
                Acount1[row-rowE:row-rowS, col-colE:col-colS] += A[rowS:rowE, colS:colE]                
                windArea[row-rowE:row-rowS, col-colE:col-colS]+=1
                
        Acount0 = windArea-Acount1
        C[ np.logical_or(Acount1 == windArea-1,Acount1 > windArea*0.65)  ] = 1   
        #C[Acount1 > windArea*0.65 ] = 1  
        C[np.logical_or(Acount0 == windArea-1, Acount0 > windArea*0.65) ] = 0            
        #C[Acount0 > windArea*0.65 ] = 0    
        
        return C
    
def hist_equal(method, img):# Contrast stretching
    new_image=np.zeros(img.shape)
    if method=='rescale_intensity':  
        p2, p98 = np.percentile(img, (0.1, 99.9))
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
        new_image=img_rescale
    elif method=='equalize_hist':
        # Equalization
        img_eq = exposure.equalize_hist(img)
        new_image=img_rescale
    elif method=='adapthist':
        # Adaptive Equalization
        img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
        new_image=img_rescale
    return new_image


def reconstructed_svd(X, n_comp):
    scaler = StandardScaler()# Fit on training set only.
    #scaler=MinMaxScaler((0,1))
    #scaler.fit(X)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set. 
    X = scaler.fit_transform(X)
    # define transform
    svd = TruncatedSVD(n_components=n_comp)#7 is the best for now
    # prepare transform on dataset
    X_svd=svd.fit_transform(X)
    Vt=svd.components_
    recostructed=X_svd.dot(Vt)
    return X_svd, recostructed

def trunc_svd(X, n_comp):
    scaler = StandardScaler()# Fit on training set only.
    #scaler=MinMaxScaler((-1,1))
    #scaler.fit(X)# Apply transform to both the training set and the test set.
    # Apply transform to both the training set and the test set. 
    X = scaler.fit_transform(X)
    # define transform
    svd = TruncatedSVD(n_components=n_comp)#7 is the best for now
    # prepare transform on dataset
    X_svd=svd.fit_transform(X)
    return X_svd