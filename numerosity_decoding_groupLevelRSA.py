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
from sklearn.preprocessing import MinMaxScaler
from os.path import join as pj

# make stimulus RDM
eventMatrix = np.array([[1,1,1,1],[1,1,1,2],[1,1,1,3],[1,1,1,4],[1,1,1,5],[1,1,1,6],[1,1,1,7],[1,1,1,8],[1,1,1,9],[1,1,1,10],
                [1,1,2,11],[1,1,2,12],[1,1,2,13],[1,1,2,14],[1,1,2,15],[1,1,2,16],[1,1,2,17],[1,1,2,18],[1,1,2,19],[1,1,2,20],
                [1,2,1,21],[1,2,1,22],[1,2,1,23],[1,2,1,24],[1,2,1,25],[1,2,1,26],[1,2,1,27],[1,2,1,28],[1,2,1,29],[1,2,1,30],
                [1,2,2,31],[1,2,2,32],[1,2,2,33],[1,2,2,34],[1,2,2,35],[1,2,2,36],[1,2,2,37],[1,2,2,38],[1,2,2,39],[1,2,2,40],
                [2,1,1,41],[2,1,1,42],[2,1,1,43],[2,1,1,44],[2,1,1,45],[2,1,1,46],[2,1,1,47],[2,1,1,48],[2,1,1,49],[2,1,1,50],
                [2,1,2,51],[2,1,2,52],[2,1,2,53],[2,1,2,54],[2,1,2,55],[2,1,2,56],[2,1,2,57],[2,1,2,58],[2,1,2,59],[2,1,2,60],
                [2,2,1,61],[2,2,1,62],[2,2,1,63],[2,2,1,64],[2,2,1,65],[2,2,1,66],[2,2,1,67],[2,2,1,68],[2,2,1,69],[2,2,1,70],
                [2,2,2,71],[2,2,2,72],[2,2,2,73],[2,2,2,74],[2,2,2,75],[2,2,2,76],[2,2,2,77],[2,2,2,78],[2,2,2,79],[2,2,2,80]])

# make correlation matrix
index = 0
numRDM = []
fsRDM = []
isRDM = []
LLFRDM = np.load('C:/Users/tclem/Desktop/MEG/LowLevelMatrix.npy') # low-level features

for x in range(80):
    for y in range(80):
        if x != y and x + y < 80: #x + y < 80:
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

subjs = ['3','4','5','6','8','9','10'] # '4','5','6',  # '2','7',
path =  r'C:\Users\tclem\Desktop\MEG'
# make 4 dimension x 2 (r value and p value) empty matrix 
subjNums, tps, RDM, fold, repeats = len(subjs), 80, 3200, 3, 100
RDMcorrNum = np.zeros((subjNums,tps))
RDMpNum = np.zeros((subjNums,tps))
RDMcorrFs = np.zeros((subjNums,tps))
RDMpFs = np.zeros((subjNums,tps))
RDMcorrIs = np.zeros((subjNums,tps))
RDMpIs = np.zeros((subjNums,tps))
RDMcorrLLF = np.zeros((subjNums,tps))
RDMpLLF = np.zeros((subjNums,tps))

partial = True
subjNum = 0
for subj in subjs:
    fileName = 'MEGRDM3x100_subj'+ subj + '.npy'
    filePath = pj(path, fileName)
    # compute partial spearman correlation, with other 2 RDM controlled 
    data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex 
    subIndex,t,re,foldIndex,RDMindex = data.shape
    data = data.reshape(subIndex*t*re*foldIndex,RDMindex)
    # normalize the MEG RDM to [0,1]
    min_max_scaler = MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    data = data.reshape(t,re,foldIndex,RDMindex)
    
    for tp in range(tps):
        datatmp = data[tp,:] # subIndex,t,re,foldIndex,RDMindex 
        RDMtmp = np.average(data[tp,:,:,:], axis=(0, 1))
        if partial == True:
            pdData = pd.DataFrame({'respRDM':RDMtmp,'numRDM':numRDM,'fsRDM':fsRDM,'isRDM':isRDM,'LLFRDM':LLFRDM})
            corr=pg.partial_corr(pdData,x='respRDM',y='numRDM',x_covar=['fsRDM','isRDM','LLFRDM'],tail='one-sided',method='spearman') 
            RDMcorrNum[subjNum,tp] = corr['r']
            RDMpNum[subjNum,tp] = corr['p-val']
        
            corr=pg.partial_corr(pdData,x='respRDM',y='fsRDM',x_covar=['numRDM','isRDM','LLFRDM'],tail='one-sided',method='spearman') 
            RDMcorrFs[subjNum,tp] = corr['r']
            RDMpFs[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='isRDM',x_covar=['numRDM','fsRDM','LLFRDM'],tail='one-sided',method='spearman') 
            RDMcorrIs[subjNum,tp] = corr['r']
            RDMpIs[subjNum,tp] = corr['p-val']
            
            corr=pg.partial_corr(pdData,x='respRDM',y='LLFRDM',x_covar=['numRDM','fsRDM','isRDM'],tail='one-sided',method='spearman') 
            RDMcorrLLF[subjNum,tp] = corr['r']
            RDMpLLF[subjNum,tp] = corr['p-val']
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
            
            corr=pg.corr(x=RDMtmp,y=LLFRDM,tail='two-sided',method='spearman') 
            RDMcorrLLF[subjNum,tp] = corr['r']
            RDMpLLF[subjNum,tp] = corr['p-val']
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
tps = 80 # time points = 80
pAvgNum = np.zeros(tps)
pAvgFs = np.zeros(tps)
pAvgIs = np.zeros(tps)
pAvgLLF = np.zeros(tps)

FDRCorrected = True
if FDRCorrected == True: # May not right
    corrAvgNum = np.average(RDMcorrNum,axis=0)
    corrAvgFs = np.average(RDMcorrFs,axis=0)
    corrAvgIs = np.average(RDMcorrIs,axis=0)
    corrAvgLLF= np.average(RDMcorrLLF,axis=0)
    for tp in range(tps):
        pAvgNum[tp] = np.average(fdr(RDMpNum[:,tp]))
        pAvgFs[tp] = np.average(fdr(RDMpFs[:,tp]))
        pAvgIs[tp] = np.average(fdr(RDMpIs[:,tp]))
        pAvgLLF[tp] = np.average(fdr(RDMpLLF[:,tp]))
elif FDRCorrected == False:
    # average cross different 
    corrAvgNum = np.average(RDMcorrNum,axis=0)
    pAvgNum = np.average(RDMpNum,axis=0)
    corrAvgFs = np.average(RDMcorrFs,axis=0)
    pAvgFs = np.average(RDMpFs,axis=0)
    corrAvgIs = np.average(RDMcorrIs,axis=0)
    pAvgIs = np.average(RDMpIs,axis=0)
    corrAvgLLF= np.average(RDMcorrLLF,axis=0)
    pAvgLLF = np.average(RDMpLLF,axis=0)

import matplotlib.pyplot as plt
plt.plot(range(-10,tps-10),corrAvgNum,label='Number',color='brown')
plt.plot(range(-10,tps-10),corrAvgFs,label='Field size',color='mediumblue')
plt.plot(range(-10,tps-10),corrAvgIs,label='Item size',color='forestgreen') #darkorange
plt.plot(range(-10,tps-10),corrAvgLLF,label='low-level feature',color='black')
# plot the significant line
pAvgNum[(pAvgNum>0.05)] = None
pAvgNum[corrAvgNum<0] = None
pAvgNum[(pAvgNum<=0.05)] = -0.15
pAvgFs[(pAvgFs>0.05)] = None
pAvgFs[corrAvgFs<0] = None
pAvgFs[(pAvgFs<=0.05)] = -0.17
pAvgIs[(pAvgIs>0.05)] = None
pAvgIs[corrAvgIs<0] = None
pAvgIs[(pAvgIs<=0.05)] = -0.19
pAvgLLF[(pAvgLLF>0.05)] = None
pAvgLLF[corrAvgLLF<0] = None
pAvgLLF[(pAvgLLF<=0.05)] = -0.21
plt.plot(range(-10,tps-10),pAvgNum,color='brown')
plt.plot(range(-10,tps-10),pAvgFs,color='mediumblue')
plt.plot(range(-10,tps-10),pAvgIs,color='forestgreen')
plt.plot(range(-10,tps-10),pAvgLLF,color='black')


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
