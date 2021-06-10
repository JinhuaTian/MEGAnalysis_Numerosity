# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:36:10 2021

@author: tclem

# compute partial Spearman correlation
# https://pingouin-stats.org/generated/pingouin.partial_corr.html
"""
import numpy as np
import pingouin as pg
import pandas as pd

filePath = r'C:\Users\tclem\Desktop\MEG\encoding3x20.npy'

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
idRDM = []

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
            # id RDM
            if eventMatrix[x,3] == eventMatrix[y,3]:
                idRDM.append(0)
            else:
                idRDM.append(1)

# compute partial spearman correlation, with other 2 RDM controlled 

data = np.load(filePath) # subj, time point, RDM, fold, repeat
# make 3 x 2 empty matrix 
subjs, tps, RDM, fold, repeats = data.shape
RDMcorrNum = np.zeros((subjs,tps))
RDMpNum = np.zeros((subjs,tps))
RDMcorrFs = np.zeros((subjs,tps))
RDMpFs = np.zeros((subjs,tps))
RDMcorrIs = np.zeros((subjs,tps))
RDMpIs = np.zeros((subjs,tps))

# select index list:
index = 0
indexlist = []
for x in range(80):
    for y in range(80):
        if x != y and x + y < 80: # exclude same id
            indexlist.append(index)
        index = index + 1

data = data[:,:,indexlist,:,:]#remove 0 component

for subj in range(subjs):
    for tp in range(tps):
        datatmp = data[subj, tp,:]
        RDMtmp = np.average(data[subj,tp,:,:,:], axis=(1, 2))
        
        pdData = pd.DataFrame({'respRDM':RDMtmp,'numRDM':numRDM,'fsRDM':fsRDM,'isRDM':isRDM})
        
        corr=pg.partial_corr(pdData,x='respRDM',y='numRDM',x_covar=['fsRDM','isRDM'],tail='two-sided',method='spearman') 
        RDMcorrNum[subj,tp] = corr['r']
        RDMpNum[subj,tp] = corr['p-val']
    
        corr=pg.partial_corr(pdData,x='respRDM',y='fsRDM',x_covar=['numRDM','isRDM'],tail='two-sided',method='spearman') 
        RDMcorrFs[subj,tp] = corr['r']
        RDMpFs[subj,tp] = corr['p-val']
        
        corr=pg.partial_corr(pdData,x='respRDM',y='isRDM',x_covar=['numRDM','fsRDM'],tail='two-sided',method='spearman') 
        RDMcorrIs[subj,tp] = corr['r']
        RDMpIs[subj,tp] = corr['p-val']

# average cross different 
corrAvgNum = np.average(RDMcorrNum,axis=0)
pAvgNum = np.average(RDMpNum,axis=0)
corrAvgFs = np.average(RDMcorrFs,axis=0)
pAvgFs = np.average(RDMpFs,axis=0)
corrAvgIs = np.average(RDMcorrIs,axis=0)
pAvgIs = np.average(RDMpIs,axis=0)


import matplotlib.pyplot as plt
plt.plot(range(-20,tps-20),corrAvgNum,label='Number',color='brown')
plt.plot(range(-20,tps-20),corrAvgFs,label='Field size',color='mediumblue')
plt.plot(range(-20,tps-20),corrAvgIs,label='Item size',color='forestgreen') #darkorange
# plot the significant line
pAvgNum[(pAvgNum>0.05)] = None
pAvgNum[(pAvgNum<=0.05)] = -0.15
pAvgFs[(pAvgFs>0.05)] = None
pAvgFs[(pAvgFs<=0.05)] = -0.17
pAvgIs[(pAvgIs>0.05)] = None
pAvgIs[(pAvgIs<=0.05)] = -0.19
plt.plot(range(-20,tps-20),pAvgNum,color='brown')
plt.plot(range(-20,tps-20),pAvgFs,color='mediumblue')
plt.plot(range(-20,tps-20),pAvgIs,color='forestgreen')


plt.xlabel('Time points(10ms)')
plt.ylabel('Partial spearman correlation')
# 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
plt.title('Time course of partial Spearman correlations between MEG RDMs and model RDMs')
plt.legend()
plt.show()





