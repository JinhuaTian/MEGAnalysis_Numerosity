# -*- coding: utf-8 -*-
"""
@author: tclem
# compute partial Spearman correlation
# https://pingouin-stats.org/generated/pingouin.partial_corr.html
"""

import numpy as np
#import pingouin as pg
from pingouin import correlation as pg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from os.path import join as pj
from neurora.stuff import clusterbased_permutation_1d_1samp_1sided as clusterP
# load stimulus RDM
eventMatrix = np.loadtxt(r'C:\Users\tclem\Documents\GitHub\MEGAnalysis_Numerosity\postProcessing\STI.txt')

# make correlation matrix
index = 0
numRDM = []
fsRDM = []
isRDM = []
shapeRDM = []
#LLFRDM = np.load('/nfs/a2/userhome/tianjinhua/workingdir/meg/LowLevelMatrix.npy') # low-level features

cc = 0
# compute model RDM
labelNum = 80
for x in range(labelNum):
    for y in range(x+1,labelNum):
        #if x != y and x + y < 80: #x + y < 80:
        cc = cc+1
        # num RDM
        if eventMatrix[x,0] == eventMatrix[y,0]:
            numRDM.append(0)
        else:
            numRDM.append(1)
        # fs RDM
        if eventMatrix[x,1] == eventMatrix[y,1]:
            fsRDM.append(0)
        else:
            fsRDM.append(1)
        # is RDM
        if eventMatrix[x,2] == eventMatrix[y,2]:
            isRDM.append(0)
        else:
            isRDM.append(1)
        # shape RDM
        if eventMatrix[x,3] == eventMatrix[y,3]:
            shapeRDM.append(0)
        else:
            shapeRDM.append(1)
        
#'004','005','006','007','009','011','012','013','014','015','016','017','018','019','020','021','022','023'
subjs = ['004','005','006','007','009','011','012','013','014','015','016','017','018','019','020','021','022','023']
path = r'E:\temp2'
# make 4 dimension x 2 (r value and p value) empty matrix 
subjNums, tps, RDM, fold, repeats = len(subjs), 240, 3160, 3, 100 # 3*80
RDMcorrNum = np.zeros((subjNums,tps))
RDMpNum = np.zeros((subjNums,tps))
RDMcorrFs = np.zeros((subjNums,tps))
RDMpFs = np.zeros((subjNums,tps))
RDMcorrIs = np.zeros((subjNums,tps))
RDMpIs = np.zeros((subjNums,tps))
RDMcorrShape = np.zeros((subjNums,tps))
RDMpShape = np.zeros((subjNums,tps))

partial = False
subjNum = 0
for subj in subjs:
    fileName = 'ctfRDM3x100x300hz_subj'+ subj + '.npy'
    filePath = pj(path, fileName)
    # compute partial spearman correlation, with other 2 RDM controlled 
    data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex 
    t,re,foldIndex,RDMindex = data.shape
    data = data.reshape(t*re*foldIndex,RDMindex)
    # normalize the MEG RDM
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    data = data.reshape(t,re,foldIndex,RDMindex)
    
    for tp in range(tps):
        datatmp = data[tp,:] # subIndex,t,re,foldIndex,RDMindex 
        RDMtmp = np.average(data[tp,:,:,:], axis=(0, 1))
        if partial == True:
            pdData = pd.DataFrame({'respRDM':RDMtmp,'numRDM':numRDM,'fsRDM':fsRDM,'isRDM':isRDM,'shapeRDM':shapeRDM})
            corr=pg.partial_corr(pdData,x='respRDM',y='numRDM',x_covar=['fsRDM','isRDM','shapeRDM'],alternative='one-sided',method='spearman') 
            RDMcorrNum[subjNum,tp] = corr['r']
            RDMpNum[subjNum,tp] = corr['p-val']
        
            corr=pg.partial_corr(pdData,x='respRDM',y='fsRDM',x_covar=['numRDM','isRDM','shapeRDM'],alternative='one-sided',method='spearman') 
            RDMcorrFs[subjNum,tp] = corr['r']
            RDMpFs[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='isRDM',x_covar=['numRDM','fsRDM','shapeRDM'],alternative='one-sided',method='spearman') 
            RDMcorrIs[subjNum,tp] = corr['r']
            RDMpIs[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='shapeRDM',x_covar=['numRDM','fsRDM','isRDM'],alternative='one-sided',method='spearman') 
            RDMcorrShape[subjNum,tp] = corr['r']
            RDMpShape[subjNum,tp] = corr['p-val']
        elif partial == False:
            corr=pg.corr(x=RDMtmp,y=numRDM,alternative='two-sided',method='spearman') 
            RDMcorrNum[subjNum,tp] = corr['r']
            RDMpNum[subjNum,tp] = corr['p-val']
        
            corr=pg.corr(x=RDMtmp,y=fsRDM,alternative='two-sided',method='spearman') 
            RDMcorrFs[subjNum,tp] = corr['r']
            RDMpFs[subjNum,tp] = corr['p-val']
            
            corr=pg.corr(x=RDMtmp,y=isRDM,alternative='two-sided',method='spearman') 
            RDMcorrIs[subjNum,tp] = corr['r']
            RDMpIs[subjNum,tp] = corr['p-val']
            
            corr=pg.corr(x=RDMtmp,y=shapeRDM,alternative='two-sided',method='spearman') 
            RDMcorrShape[subjNum,tp] = corr['r']
            RDMpShape[subjNum,tp] = corr['p-val']
    subjNum = subjNum + 1
    del data

from statsmodels.stats.multitest import fdrcorrection
#fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)
def fdr(x):
    xx = fdrcorrection(x, alpha=0.05, method='indep', is_sorted=False)
    return xx

tps = 240 # time points = 80
pAvgNum = np.zeros(tps)
pAvgFs = np.zeros(tps)
pAvgIs = np.zeros(tps)
pAvgShape = np.zeros(tps)

# correct the p value
clusterCorrected = True #clusterP
FDRCorrected =False
noCorrection=False
if clusterCorrected == True:
    pAvgNum = clusterP(RDMpNum)
    pAvgFs = clusterP(RDMpFs)
    pAvgIs = clusterP(RDMpIs)
    pAvgShape = clusterP(RDMpShape)
elif FDRCorrected == True:
    for tp in range(tps):
        pAvgNum[tp] = np.average(fdr(RDMpNum[:,tp]))
        pAvgFs[tp] = np.average(fdr(RDMpFs[:,tp]))
        pAvgIs[tp] = np.average(fdr(RDMpIs[:,tp]))
        pAvgShape[tp] = np.average(fdr(RDMpShape[:,tp]))
elif noCorrection == True:
    # average cross different subjects
    corrAvgNum = np.average(RDMcorrNum,axis=0)
    pAvgNum = np.average(RDMpNum,axis=0)
    corrAvgFs = np.average(RDMcorrFs,axis=0)
    pAvgFs = np.average(RDMpFs,axis=0)
    corrAvgIs = np.average(RDMcorrIs,axis=0)
    pAvgIs = np.average(RDMpIs,axis=0)
    corrAvgShape= np.average(RDMcorrShape,axis=0)
    pAvgShape = np.average(RDMpShape,axis=0)

corrAvgNum = np.average(RDMcorrNum,axis=0)
corrAvgFs = np.average(RDMcorrFs,axis=0)
corrAvgIs = np.average(RDMcorrIs,axis=0)
corrAvgShape = np.average(RDMcorrShape,axis=0)
    
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9,6),dpi=100)
plt.plot((np.arange(-30,tps-30))/3,corrAvgNum,label='Number',color='brown')
plt.plot((np.arange(-30,tps-30))/3,corrAvgFs,label='Field size',color='mediumblue')
plt.plot((np.arange(-30,tps-30))/3,corrAvgIs,label='Item size',color='forestgreen') #darkorange
plt.plot((np.arange(-30,tps-30))/3,corrAvgShape,label='Shape',color='black')
# plot the significant line
th = 0.05
thCorr = np.min([np.min(corrAvgNum),np.min(corrAvgFs),np.min(corrAvgIs),np.min(corrAvgShape)])

pAvgNum[(pAvgNum>th)] = None
pAvgNum[corrAvgNum<0] = None
pAvgNum[(pAvgNum<=th)] = thCorr-0.01
pAvgFs[(pAvgFs>th)] = None
pAvgFs[corrAvgFs<0] = None
pAvgFs[(pAvgFs<=th)] =  thCorr-0.02
pAvgIs[(pAvgIs>th)] = None
pAvgIs[corrAvgIs<0] = None
pAvgIs[(pAvgIs<=th)] =  thCorr-0.03
pAvgShape[(pAvgShape>th)] = None
pAvgShape[corrAvgShape<0] = None
pAvgShape[(pAvgShape<=th)] =  thCorr-0.04
plt.plot((np.arange(-30,tps-30))/3,pAvgNum,color='brown') # range(-30,tps-30)
plt.plot((np.arange(-30,tps-30))/3,pAvgFs,color='mediumblue')
plt.plot((np.arange(-30,tps-30))/3,pAvgIs,color='forestgreen')
plt.plot((np.arange(-30,tps-30))/3,pAvgShape,color='black')

plt.xlabel('Time points(10ms)')
if partial == True:
    plt.ylabel('Partial spearman correlation')
    # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
    plt.title('Time course of partial Spearman correlations between MEG RDMs and model RDMs')
elif partial == False:
    plt.ylabel('Spearman correlation')
    # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
    plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
plt.legend()
plt.show()

'''
# plot average acc
partialAvgAcc = np.average(data, axis=(2, 3, 4))

import matplotlib.pyplot as plt
partialAvgAcc = np.squeeze(partialAvgAcc)
x = partialAvgAcc.shape
plt.plot(range(-30,x[0]-30),partialAvgAcc)
plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.title('Pairwise decoding accuracy(average)')
'''

#peak detecttion:
NumOnsetTp = np.zeros(len(subjs))
NumPeakTp = np.zeros(len(subjs))
FsOnsetTp = np.zeros(len(subjs))
FsPeakTp = np.zeros(len(subjs))
for sub in range(len(subjs)): #50~200ms, 15~30 * 3 tps
    tp = np.where(RDMpNum[sub, 30:90]<0.05)
    NumOnsetTp[sub] = tp[0][0]
    NumPeakTp[sub] = np.where(RDMcorrNum[sub, 45:90] == np.max(RDMcorrNum[sub, 45:90]))[0]
    
    tp = np.where(RDMpFs[sub, 30:90]<0.05)
    FsOnsetTp[sub] = tp[0][0]
    FsPeakTp[sub] = np.where(RDMcorrFs[sub, 45:90] == np.max(RDMcorrFs[sub, 45:90]))[0]

NumOnsetTp = NumOnsetTp *3 # + 50
NumPeakTp = NumPeakTp *3 + 50
FsOnsetTp = FsOnsetTp *3 # + 50
FsPeakTp = FsPeakTp *3 + 50

import pandas as pd
import seaborn as sns
dataOnset = pd.DataFrame({"Number onset":NumOnsetTp,"Field size onset":FsOnsetTp})
sns.boxplot(data=dataOnset)
plt.title('Onset time difference')
plt.ylabel('Latency(ms)')
plt.legend()
plt.show()

dataPeak = pd.DataFrame({"Number peak":NumPeakTp,"Field size peak":FsPeakTp})
sns.boxplot(data=dataPeak)
plt.title('Peak difference')
plt.ylabel('Latency(ms)')
plt.legend()
plt.show()

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

# permutation test
_,_,p = permutation_diff(NumPeakTp,FsPeakTp)
_,_,p2 = permutation_diff(NumOnsetTp,FsOnsetTp)







