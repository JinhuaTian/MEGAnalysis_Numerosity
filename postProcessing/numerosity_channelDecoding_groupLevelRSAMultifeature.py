# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 20:44:26 2021

@author: tclem
"""
import numpy as np
#import pingouin as pg
from pingouin import correlation as pg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from os.path import join as pj
from neurora.stuff import clusterbased_permutation_1d_1samp_1sided as clusterP
# load stimulus RDM

modelRDM = np.load('E:/temp2/ModelRDM_NumFsIsShapeTfaDenLLF.npy') 
# make correlation matrix
RDMName = ['Number','Field size','Item size','Shape','Total field area', 'Density', 'Low-level Feature']

#'004','005','006','007','009','011','012','013','014','015','016','017','018','019','020','021','022','023'
subjs = ['004','005','006','007','009','011','012','013','014','015','016','017','018','019','020','021','022','023']
path = r'E:\temp2'
# make 4 dimension x 2 (r value and p value) empty matrix 
subjNums, tps, RDM, fold, repeats = len(subjs), 240, 3160, 3, 100 # 3*80

for i in range(len(RDMName)):
    exec('RDMcorr{} = np.zeros((subjNums,tps))'.format(i))
    exec('RDMp{} = np.zeros((subjNums,tps))'.format(i))
    exec('ModelRDM{} = modelRDM[{},:]'.format(i,i))

del modelRDM

side = 'two-sided'
partial = True

subjNum = 0
for subj in subjs:
    fileName = 'ctfRDM3x100x300hz_subj'+ subj + '.npy'
    filePath = pj(path, fileName)
    # compute partial spearman correlation, with other 2 RDM controlled 
    data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex 
    t,re,foldIndex,RDMindex = data.shape
    data = data.reshape(t*re*foldIndex,RDMindex)
    # normalize the MEG RDM
    #  scaler = StandardScaler()
    #  data = scaler.fit_transform(data)
    data = data.reshape(t,re,foldIndex,RDMindex)
    
    for tp in range(tps):
        datatmp = data[tp,:] # subIndex,t,re,foldIndex,RDMindex 
        RDMtmp = np.average(data[tp,:,:,:], axis=(0, 1))
        if partial == True: # 'RDM0','RDM1','RDM2','RDM3','RDM4','RDM5','RDM6'
            #'numRDM','fsRDM','isRDM','shapeRDM','tfaRDM','denRDM','denRDM'
            pdData = pd.DataFrame({'respRDM':RDMtmp,'RDM0':ModelRDM0,'RDM1':ModelRDM1,'RDM2':ModelRDM2,'RDM3':ModelRDM3,
                                   'RDM4':ModelRDM4,'RDM5':ModelRDM5,'RDM6':ModelRDM6})
            corr=pg.partial_corr(pdData,x='respRDM',y='RDM0',x_covar=['RDM1','RDM2','RDM3','RDM4','RDM5','RDM6'],alternative=side,method='spearman') 
            RDMcorr0[subjNum,tp] = corr['r']
            RDMp0[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='RDM1',x_covar=['RDM0','RDM2','RDM3','RDM4','RDM5','RDM6'],alternative=side, method='spearman') 
            RDMcorr1[subjNum,tp] = corr['r']
            RDMp1[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='RDM2',x_covar=['RDM0','RDM1','RDM3','RDM4','RDM5','RDM6'],alternative=side, method='spearman') 
            RDMcorr2[subjNum,tp] = corr['r']
            RDMp2[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='RDM3',x_covar=['RDM0','RDM1','RDM2','RDM4','RDM5','RDM6'],alternative=side, method='spearman') 
            RDMcorr3[subjNum,tp] = corr['r']
            RDMp3[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='RDM4',x_covar=['RDM0','RDM1','RDM2','RDM3','RDM5','RDM6'],alternative=side, method='spearman') 
            RDMcorr4[subjNum,tp] = corr['r']
            RDMp4[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='RDM5',x_covar=['RDM0','RDM1','RDM2','RDM3','RDM4','RDM6'],alternative=side, method='spearman') 
            RDMcorr5[subjNum,tp] = corr['r']
            RDMp5[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='RDM6',x_covar=['RDM0','RDM1','RDM2','RDM3','RDM4','RDM5'],alternative=side, method='spearman') 
            RDMcorr6[subjNum,tp] = corr['r']
            RDMp6[subjNum,tp] = corr['p-val']
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

clusterCorrected = True #clusterP
FDRCorrected =False
noCorrection=False
for i in range(len(RDMName)):
    exec('avgCorr{} =np.average(RDMcorr{},axis=0)'.format(i,i))
    if clusterCorrected == True:
        exec('avgP{} = clusterP(RDMp{})'.format(i,i))

'''
elif FDRCorrected == True:
    for tp in range(tps):
        break
        #pAvgNum[tp] = np.average(fdr(RDMpNum[:,tp]))
elif noCorrection == True:
    break
    # average cross different subjects
    # pAvgNum = np.average(RDMpNum,axis=0)
'''

color = ["Red", "Purple", "Gray",  "Blue", "Green", "Orange",  'brown']


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9,6),dpi=100)
th = 0.05
thCorr = np.min([np.min(avgCorr0),np.min(avgCorr1),np.min(avgCorr2),np.min(avgCorr3),np.min(avgCorr4),np.min(avgCorr5),np.min(avgCorr6)])
thMinus = 0.01
for i in range(len(RDMName)):
    exec('plt.plot((np.arange(-30,tps-30))/3,avgCorr{},label=RDMName[i],color=color[i])'.format(i))  
    # judge whether there are significant line
    if not exec('(np.array(avgP{})<1).all()'.format(i)):
        # plot the significant line
        exec('avgP{}[(avgP{}>th)] = None'.format(i,i))
        exec('avgP{}[avgP{}<0] = None'.format(i,i))
        exec('avgP{}[(avgP{}<=th)] = thCorr-thMinus'.format(i,i))
        exec('plt.plot((np.arange(-30,tps-30))/3,avgP{},color=color[i])'.format(i)) # range(-30,tps-30)
        thMinus = thMinus + 0.01

plt.xlabel('Time points(10ms)')
if partial == False:
    plt.ylabel('Partial spearman correlation')
    # 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
    plt.title('Time course of partial Spearman correlations between MEG RDMs and model RDMs')
elif partial == True:
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

'''
#peak detecttion:
NumOnsetTp = np.zeros(len(subjs))
NumPeakTp = np.zeros(len(subjs))
FsOnsetTp = np.zeros(len(subjs))
FsPeakTp = np.zeros(len(subjs))
for sub in range(len(subjs)): #50~200ms, 15~30 * 3 tps
    tp = np.where(RDMpNum[sub, 45:90]<0.05)
    NumOnsetTp[sub] = tp[0][0]
    NumPeakTp[sub] = np.where(RDMcorrNum[sub, 45:90] == np.max(RDMcorrNum[sub, 45:90]))[0]
    
    tp = np.where(RDMpFs[sub, 45:90]<0.05)
    FsOnsetTp[sub] = tp[0][0]
    FsPeakTp[sub] = np.where(RDMcorrFs[sub, 45:90] == np.max(RDMcorrFs[sub, 45:90]))[0]

NumOnsetTp = NumOnsetTp *3 + 50
NumPeakTp = NumPeakTp *3 + 50
FsOnsetTp = FsOnsetTp *3 + 50
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
'''