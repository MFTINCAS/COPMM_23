# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:10:52 2021

@author: Cristiano
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
from PIL import Image
from sklearn.metrics import roc_auc_score

def metrics_result(y_true,y_pred_gmm, y_pred_cop):
    metrics_cop=[]
    metrics_gmm=[]
    #plt.imshow(y_true.reshape(390,200), cmap=plt.cm.gray)
    #plt.imshow(y_pred_gmm.reshape(390,200), cmap=plt.cm.gray)
    #plt.imshow(y_pred_cop.reshape(390,200), cmap=plt.cm.gray)
    
    #A=confusion_matrix(y_true, y_pred_gmm)
    #B=confusion_matrix(y_true, y_pred_cop)
    M_g = matthews_corrcoef(y_true,y_pred_gmm)
    M_my= matthews_corrcoef(y_true,y_pred_cop)
    M_g = (M_g+1.0)/2.0  # così abbiamo i valori tra 0 e 1
    M_my= (M_my+1.0)/2.0
    F_g = f1_score(y_true,y_pred_gmm)
    F_my= f1_score(y_true,y_pred_cop)
    acc_g=accuracy_score(y_true, y_pred_gmm)
    acc_my=accuracy_score(y_true, y_pred_cop)
    metrics_cop.append([M_my,F_my,acc_my])
    metrics_gmm.append([M_g,F_g,acc_g])
    
    auc=roc_auc_score(y_true, y_pred_cop)
    print('metrics for gmm:', 'matthews:' ,M_g, 'f1', F_g, 'oa',acc_g)
    print('metrics for copulamixture:', 'matthews', M_my,'f1',F_my, 'oa',acc_my)
    
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_cop).ravel()
    return metrics_cop, metrics_gmm
    
# def split_random(len_dataset, w):
#     len_dataset=np.arange(len_dataset)
#     len_dataset=sorted(len_dataset, key=lambda len_dataset: random.random())
#     a=np.array_split(len_dataset,w)
#     return a