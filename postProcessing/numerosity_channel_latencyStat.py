# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 22:13:40 2021

@author: tclem
"""
import numpy as np
from os.path import join as pj

rootPath = 'E:/temp2'
data = np.load(pj(rootPath,'peaks_4RDM.npy'))

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

# ['Number','Field size','Item size','Shape','Total field area', 'Density', 'Low-level Feature']
pValue = []
pMatrix = np.zeros((data.shape[0],data.shape[0]))
'''
reorgan = [6,0,1,4,5,2,3]
countx = 0
for i in reorgan:
    county = countx
    for j in reorgan[countx+1:]:
        _,_,pp = permutation_diff(data[i,:], data[j,:])
        pValue.append(pp)
        pMatrix[countx+1,county] = pp
        county = county + 1
    countx = countx + 1
print(pp)
'''
countx = 0
for i in range(4):
    county = countx
    for j in range(countx+1,4):
        _,_,pp = permutation_diff(data[i,:], data[j,:])
        pValue.append(pp)
        pMatrix[countx+1,county] = pp
        county = county + 1
    countx = countx + 1
