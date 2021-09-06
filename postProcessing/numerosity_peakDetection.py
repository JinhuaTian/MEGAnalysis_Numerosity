# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:44:02 2021

@author: tclem

scripts for correlation peak detection
"""

import numpy as np
#import pingouin as pg
from pingouin import correlation as pg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from os.path import join as pj

# load stimulus RDM
eventMatrix = np.loadtxt('C:/Users/tclem/Documents/GitHub/MEGAnalysis_Numerosity/postProcessing/STI.txt')

# make correlation matrix
index = 0
numRDM = []
fsRDM = []
isRDM = []
shapeRDM = []
#LLFRDM = np.load('/nfs/a2/userhome/tianjinhua/workingdir/meg/LowLevelMatrix.npy') # low-level features

cc = 0
# compute model RDM
for x in range(80):
    for y in range(80):
        if x != y and x + y < 80: #x + y < 80:
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
        

subjs = ['004','005','006','007','009','011','012','013','014','015','016','017','018','019','020','021','022','023'] # '2','3','4','5','6','8','9','10'
path = 'E:/temp'
# make 4 dimension x 2 (r value and p value) empty matrix 
subjNums, tps, RDM, fold, repeats = len(subjs), 240, 3200, 3, 100 # 3*80
RDMcorrNum = np.zeros((subjNums,tps))
RDMpNum = np.zeros((subjNums,tps))
RDMcorrFs = np.zeros((subjNums,tps))
RDMpFs = np.zeros((subjNums,tps))
RDMcorrIs = np.zeros((subjNums,tps))
RDMpIs = np.zeros((subjNums,tps))
RDMcorrShape = np.zeros((subjNums,tps))
RDMpShape = np.zeros((subjNums,tps))

partial = True
subjNum = 0
for subj in subjs:
    fileName = 'ctfRDM3x100x300hz_subj'+ subj + '.npy'
    filePath = pj(path, fileName)
    # compute partial spearman correlation, with other 2 RDM controlled 
    data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex 
    t,re,foldIndex,RDMindex = data.shape
    data = data.reshape(t*re*foldIndex,RDMindex)
    # normalize the MEG RDM to [0,1]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = data.reshape(t,re,foldIndex,RDMindex)
    
    for tp in range(tps):
        datatmp = data[tp,:] # subIndex,t,re,foldIndex,RDMindex 
        RDMtmp = np.average(data[tp,:,:,:], axis=(0, 1))
        if partial == True:
            pdData = pd.DataFrame({'respRDM':RDMtmp,'numRDM':numRDM,'fsRDM':fsRDM,'isRDM':isRDM,'shapeRDM':shapeRDM})
            corr=pg.partial_corr(pdData,x='respRDM',y='numRDM',x_covar=['fsRDM','isRDM','shapeRDM'],tail='one-sided',method='spearman') 
            RDMcorrNum[subjNum,tp] = corr['r']
            RDMpNum[subjNum,tp] = corr['p-val']
        
            corr=pg.partial_corr(pdData,x='respRDM',y='fsRDM',x_covar=['numRDM','isRDM','shapeRDM'],tail='one-sided',method='spearman') 
            RDMcorrFs[subjNum,tp] = corr['r']
            RDMpFs[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='isRDM',x_covar=['numRDM','fsRDM','shapeRDM'],tail='one-sided',method='spearman') 
            RDMcorrIs[subjNum,tp] = corr['r']
            RDMpIs[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='shapeRDM',x_covar=['numRDM','fsRDM','isRDM'],tail='one-sided',method='spearman') 
            RDMcorrShape[subjNum,tp] = corr['r']
            RDMpShape[subjNum,tp] = corr['p-val']
        elif partial == False:
            corr=pg.corr(x=RDMtmp,y=numRDM,tail='two-sided',method='spearman') 
            RDMcorrNum[subjNum,tp] = corr['r']
            RDMpNum[subjNum,tp] = corr['p-val']
        
            corr=pg.corr(x=RDMtmp,y=fsRDM,tail='two-sided',method='spearman') 
            RDMcorrFs[subjNum,tp] = corr['r']
            RDMpFs[subjNum,tp] = corr['p-val']
            
            corr=pg.corr(x=RDMtmp,y=isRDM,tail='two-sided',method='spearman') 
            RDMcorrIs[subjNum,tp] = corr['r']
            RDMpIs[subjNum,tp] = corr['p-val']
            
            corr=pg.corr(x=RDMtmp,y=shapeRDM,tail='two-sided',method='spearman') 
            RDMcorrShape[subjNum,tp] = corr['r']
            RDMpShape[subjNum,tp] = corr['p-val']
        '''
        corr=pg.partial_corr(pdData,x='respRDM',y='numRDM',x_covar=['fsRDM','isRDM','LLFRDM'],tail='two-sided',method='spearman') 
        RDMcorrNum[subjNum,tp] = corr['r']
        RDMpNum[subjNum,tp] = corr['p-val']
    
        corr=pg.partial_corr(pdData,x='respRDM',y='fsRDM',x_covar=['numRDM','isRDM','LLFRDM'],tail='two-sided',method='spearman') 
        RDMcorrFs[subjNum,tp] = corr['r']
        RDMpFs[subjNum,tp] = corr['p-val']
        
        corr=pg.partial_corr(pdData,x='respRDM',y='isRDM',x_covar=['numRDM','fsRDM','LLFRDM'],tail='two-sided',method='spearman') 
        RDMcorrIs[subjNum,tp] = corr['r']
        RDMpIs[subjNum,tp] = corr['p-val']
        
        corr=pg.partial_corr(pdData,x='respRDM',y='LLFRDM',x_covar=['numRDM','fsRDM','isRDM'],tail='two-sided',method='spearman') 
        RDMcorrLLF[subjNum,tp] = corr['r']
        RDMpLLF[subjNum,tp] = corr['p-val']
        '''
    subjNum = subjNum + 1
    del data

from statsmodels.stats.multitest import fdrcorrection
#fdrcorrection(pvals, alpha=0.05, method='indep', is_sorted=False)
def fdr(x):
    xx = fdrcorrection(x, alpha=0.05, method='indep', is_sorted=False)
    return(xx)
tps = 240 # time points = 80
pAvgNum = np.zeros(tps)
pAvgFs = np.zeros(tps)
pAvgIs = np.zeros(tps)
pAvgShape = np.zeros(tps)

FDRCorrected = False
if FDRCorrected == True: # May not right
    corrAvgNum = np.average(RDMcorrNum,axis=0)
    corrAvgFs = np.average(RDMcorrFs,axis=0)
    corrAvgIs = np.average(RDMcorrIs,axis=0)
    corrAvgShape = np.average(RDMcorrShape,axis=0)
    for tp in range(tps):
        pAvgNum[tp] = np.average(fdr(RDMpNum[:,tp]))
        pAvgFs[tp] = np.average(fdr(RDMpFs[:,tp]))
        pAvgIs[tp] = np.average(fdr(RDMpIs[:,tp]))
        pAvgShape[tp] = np.average(fdr(RDMpShape[:,tp]))
elif FDRCorrected == False:
    # average cross different 
    corrAvgNum = np.average(RDMcorrNum,axis=0)
    pAvgNum = np.average(RDMpNum,axis=0)
    corrAvgFs = np.average(RDMcorrFs,axis=0)
    pAvgFs = np.average(RDMpFs,axis=0)
    corrAvgIs = np.average(RDMcorrIs,axis=0)
    pAvgIs = np.average(RDMpIs,axis=0)
    corrAvgShape= np.average(RDMcorrShape,axis=0)
    pAvgShape = np.average(RDMpShape,axis=0)

import matplotlib.pyplot as plt
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
plt.plot(range(-10,x[0]-10),partialAvgAcc)
plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.title('Pairwise decoding accuracy(average)')
'''
