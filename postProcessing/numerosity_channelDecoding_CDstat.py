# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:34:48 2021

@author: tclem
"""
import numpy as np
from os.path import join as pj
rootDir = r'E:\temp2\crossDecoding'

subjs = ['004','005','006','007','009','011','012','013','014','015','016','017','018','019','020','021','022','023'] 

allData = []
for subj in subjs:
    fileName = 'crossDecoding15x300hz_subj'+ subj + '.npy'
    filePath = pj(rootDir, fileName)
    
    data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex
    allData.append(data)

allData = np.stack(allData,axis=0)
avgData = np.average(allData,axis=1)


peaks = np.zeros((18,4))
'''
for subj in range(18):
    for rdm in range(4):
        peaks[subj,rdm] = np.argmax(avgData[subj,rdm,30:90])
'''

import scipy.stats as st
def ClusterSignTest(results,
                    level=0,
                    p_threshold=0.05,
                    clusterp_threshold=0.05,
                    n_threshold=2,
                    iter=1000):
    # results should be like (n_subj, n_times)
    nsubs, x = np.shape(results)

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        ts[t], p = st.wilcoxon(results[:, t]-level, zero_method='wilcox',
                               correction=True, alternative='greater', mode='auto')

        ps[t] = 1 if (p < p_threshold and ts[t] > 0) else 0

    x = np.shape(ps)[0]
    b = np.zeros([x+2])
    b[1:x+1] = ps

    index_v = np.zeros([x])

    index_n = 0
    for i in range(x):
        if b[i+1] == 1 and b[i] == 0 and b[i+2] == 1:
            index_n = index_n + 1
        if b[i+1] == 1:
            if b[i] != 0 or b[i+2] != 0:
                index_v[i] = index_n

    cluster_index, cluster_n = index_v, index_n

    if cluster_n != 0:
        cluster_ts = np.zeros([cluster_n])
        for i in range(cluster_n):
            for t in range(x):
                if cluster_index[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        # print("\nCluster-based permutation sign test")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n])
            for j in range(cluster_n):
                for t in range(x):
                    if cluster_index[t] == j + 1:
                        v = np.hstack((results[:, t], chance))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + \
                            st.wilcoxon(v1, v2,
                                        zero_method='wilcox',
                                        correction=True,
                                        alternative='greater',
                                        mode='approx')[0]
            permu_ts[i] = np.max(permu_cluster_ts)


        for i in range(cluster_n):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-clusterp_threshold):
                for t in range(x):
                    if cluster_index[t] == i + 1:
                        ps[t] = 0

    newps = np.zeros([x + 2])
    newps[1:x + 1] = ps

    for i in range(x):
        if newps[i + 1] == 1 and newps[i] != 1:
            index = 0
            while newps[i + 1 + index] == 1:
                index = index + 1
            if index < n_threshold:
                newps[i + 1:i + 1 + index] = 0

    ps = newps[1:x + 1]
    onset = (ps != 0).argmax()
    #peak = np.mean(b, axis=0).argmax()
    peak = np.argmax(ts)
    return ps, onset, peak        

# detect onset latency
onset = np.zeros((18,4))
for subj in range(18):
    for rdm in range(4):
        _,onset[subj,rdm],peaks[subj,rdm] = ClusterSignTest(allData[subj,:,rdm,30:90],level=0.5,n_threshold=3) # time window 0~200ms
        print('onset is {}'.format(onset[subj,rdm]))

# plot average 
avgData4 = np.average(avgData,axis=0)
#avgData4 = avgData4 + 60
import matplotlib.pyplot as plt
tps = 240
plt.plot((np.arange(-30,tps-30))/3,avgData4[0],label='Number',color='brown')
plt.plot((np.arange(-30,tps-30))/3,avgData4[1],label='Field size',color='mediumblue')
plt.plot((np.arange(-30,tps-30))/3,avgData4[2],label='Item size',color='forestgreen')
plt.plot((np.arange(-30,tps-30))/3,avgData4[3],label='Shape',color='black')

plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.legend()
plt.title('Cross decoding accuracy(average)')

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

import pandas as pd
import seaborn as sns

peaks = peaks*10/3
fig3 = plt.figure(figsize=(9,6),dpi=100)
dataPeak = pd.DataFrame({'Number':peaks[:,0],'Field area':peaks[:,1],'Item area':peaks[:,2],'Shape':peaks[:,3]})
#sns.boxplot(data=dataOnset)
sns.violinplot(data=dataPeak)
plt.ylim((0,250))
plt.title('Peak latency difference')
plt.ylabel('Latency(ms)')
plt.legend()
# plt.savefig(pj(rootDir,'OnsetDifference_4feature.png'))
plt.show()

onset = onset*10/3
fig3 = plt.figure(figsize=(9,6),dpi=100)
dataOnset = pd.DataFrame({'Number':onset[:,0],'Field area':onset[:,1],'Item area':onset[:,2],'Shape':onset[:,3]})
#sns.boxplot(data=dataOnset)
sns.violinplot(data=dataOnset)
plt.ylim((0,250))
plt.title('Onset latency difference')
plt.ylabel('Latency(ms)')
plt.legend()
# plt.savefig(pj(rootDir,'OnsetDifference_4feature.png'))
plt.show()

def excludeOutlier(data):
    std = np.std(data) 
    mean = np.average(data)
    newData = data[(data<(mean+3*std)) | (data>(mean-3*std))]
    return newData

pMatrix = np.zeros((4,4))
pMatrix2 = np.zeros((4,4))

for i in range(4):
    for j in range(i+1,4):
        _,_,pp = permutation_diff(excludeOutlier(peaks[:,i]), excludeOutlier(peaks[:,j]))
        pMatrix[i,j] = pp

for i in range(4):
    for j in range(i+1,4):
        _,_,pp = permutation_diff(excludeOutlier(onset[:,i]), excludeOutlier(onset[:,j]))
        pMatrix2[i,j] = pp