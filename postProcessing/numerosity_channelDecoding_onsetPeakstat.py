# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:02:11 2021

@author: tclem
"""
import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from pandas.core.frame import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join as pj

rootDir = r'E:\temp2\RSAstat'
'''
distribData = []
sumData = []
for i in range(5):
    file1 = pj(rootDir,'distrib_bs'+str(i)+'.npy')
    data1 = np.load(file1)
    distribData.append(data1)
    file2 = pj(rootDir,'summary_bs'+str(i)+'.npy')
    data2 = np.load(file2)
    sumData.append(data2)
'''
    
distribData = np.load(pj(rootDir,'distrib_bsAll.npy'))
sumData = np.load(pj(rootDir,'summary_bsAll.npy'))

dd = sumData[:,1,:]

# stastic test 
# main function of permutation
def permutation_diff(list1, list2, n_permutation=10000, tail='both'):
    """
    Make permutation test for the difference of mean values between list1 and list2
    Parameters:
    -----------
    list1, list2: two lists contain data
    n_permutation: permutation times
    tail: 'larger', one-tailed test, test if list_diff is larger than diff_scores
          'smaller', one-tailed test, test if list_diff is smaller than diff_score
          'both', two_tailed test
    Output:
    -------
    list_diff: difference between list1 and list2
    diff_scores: different values after the permutation
    pvalue: pvalues
    Examples:
    ----------
    >>> list_diff, diff_scores, pvalue = permutation_diff(list1, list2)
    """
    if not isinstance(list1, list):
        list1 = list(list1.flatten())
    if not isinstance(list2, list):
        list2 = list(list2.flatten())
    list_diff = np.nanmean(list1) - np.nanmean(list2)
    list1_len = len(list1)
    list2_len = len(list2)
    list_total = np.array(list1+list2)
    list_total_len = len(list_total)
    diff_scores = []
    for i in range(n_permutation):
        list1_perm_idx = np.sort(np.random.choice(range(list_total_len), list1_len, replace=False))
        list2_perm_idx = np.sort(list(set(range(list_total_len)).difference(set(list1_perm_idx))))
        list1_perm = list_total[list1_perm_idx]
        list2_perm = list_total[list2_perm_idx]
        diff_scores.append(np.nanmean(list1_perm) - np.nanmean(list2_perm))
    if tail == 'larger':
        pvalue = 1.0*(np.sum(diff_scores>list_diff)+1)/(n_permutation+1)
    elif tail == 'smaller':
        pvalue = 1.0*(np.sum(diff_scores<list_diff)+1)/(n_permutation+1)
    elif tail == 'both':
        pvalue = 1.0*(np.sum(np.abs(diff_scores)>np.abs(list_diff))+1)/(n_permutation+1)
    else:
        raise Exception('Wrong paramters')
    return list_diff, diff_scores, pvalue


onsetMatrix = np.zeros((4,4))
peakMatrix = np.zeros((4,4))
def excludeOutlier(data):
    std = np.std(data) 
    mean = np.average(data)
    newData = data[(data<(mean+3*std)) | (data>(mean-3*std))]
    return newData
    
    
for i in range(4):
    for j in range(i+1,4):
        _,_,pp = permutation_diff(excludeOutlier(distribData[i,0,:]), excludeOutlier(distribData[j,0,:]))
        onsetMatrix[i,j] = pp
for i in range(4):
    for j in range(i+1,4):
        _,_,pp = permutation_diff(excludeOutlier(distribData[i,1,:]), excludeOutlier(distribData[j,1,:]))
        peakMatrix[i,j] = pp

import matplotlib.pyplot as plt
plt.hist(x=distribData[0,1,:])
plt.hist(x=distribData[1,1,:])
plt.hist(x=distribData[2,1,:])
plt.hist(x=distribData[3,1,:])
plt.xlabel('latency')
plt.ylabel('count')