# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:38:05 2021

@author: Clemens
"""
import numpy as np
#import pingouin as pg
from pingouin import correlation as pg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
import matplotlib
from os.path import join as pj
#matplotlib.use('Qt5Agg')
filePath = 'D:/MEG/source'

# make stimulus RDM
eventMatrix =  np.loadtxt(r'C:\Users\Clemens\Documents\GitHub\MEGAnalysis_Numerosity\postProcessing\STI.txt')

rootDir = '/home/jhtian/workingdir/meg2/'
subjid = 'subj004'
labeltag = 'V3AB'

fileName = pj(filePath,'ctfRDM3x100_subj004' + labeltag+'.npy')

# make correlation matrix
index = 0
numRDM = []
fsRDM = []
isRDM = []
shapeRDM = []
#LLFRDM = np.load('C:/Users/tclem/Desktop/MEG/LowLevelMatrix.npy') # low-level features

labelNum = 80
# compute model RDM
for x in range(labelNum):
    for y in range(x+1,labelNum):
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


# compute partial spearman correlation, with other 2 RDM controlled 
data = np.load(fileName) # subIndex,t,re,foldIndex,RDMindex 
data = np.squeeze(data)

t,re,foldIndex,RDMindex = data.shape
data = data.reshape(t*re*foldIndex,RDMindex)
# normalize the MEG RDM to [0,1]
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = data.reshape(t,re,foldIndex,RDMindex)

# make 3 x 2 empty matrix 
tps, RDM, fold, repeats = data.shape
RDMcorrNum = np.zeros(tps)
RDMpNum = np.zeros(tps)
RDMcorrFs = np.zeros(tps)
RDMpFs = np.zeros(tps)
RDMcorrIs = np.zeros(tps)
RDMpIs = np.zeros(tps)
RDMcorrLLF = np.zeros(tps)
RDMpLLF = np.zeros(tps)
RDMcorrShape = np.zeros(tps)
RDMpShape = np.zeros(tps)
'''
# select index list:
index = 0
indexlist = []
for x in range(80):
    for y in range(80):
        if x != y and x + y < 80: # exclude same id
            indexlist.append(index)
        index = index + 1

data = data[:,:,indexlist,:,:]#remove 0 component
'''
#partial = False
method = 'Spearman'

for tp in range(tps):
    # datatmp = data[subj, tp,:] # subIndex,t,re,foldIndex,RDMindex 
    RDMtmp = np.average(data[tp,:,:,:], axis=(0, 1))
    
    pdData = pd.DataFrame({'respRDM':RDMtmp,'numRDM':numRDM,'fsRDM':fsRDM,'isRDM':isRDM,'shapeRDM':shapeRDM})
    if method == 'Spearman':
        corr=pg.partial_corr(pdData,x='respRDM',y='numRDM',x_covar=['fsRDM','isRDM','shapeRDM'],alternative='two-sided',method='spearman') 
        RDMcorrNum[tp] = corr['r']
        RDMpNum[tp] = corr['p-val']
    
        corr=pg.partial_corr(pdData,x='respRDM',y='fsRDM',x_covar=['numRDM','isRDM','shapeRDM'],alternative='two-sided',method='spearman') 
        RDMcorrFs[tp] = corr['r']
        RDMpFs[tp] = corr['p-val']
        
        corr=pg.partial_corr(pdData,x='respRDM',y='isRDM',x_covar=['numRDM','fsRDM','shapeRDM'],alternative='two-sided',method='spearman') 
        RDMcorrIs[tp] = corr['r']
        RDMpIs[tp] = corr['p-val']
        '''
        corr=pg.partial_corr(pdData,x='respRDM',y='LLFRDM',x_covar=['numRDM','fsRDM','isRDM','shapeRDM'],tail='two-sided',method='spearman') 
        RDMcorrLLF[subj,tp] = corr['r']
        RDMpLLF[subj,tp] = corr['p-val']
        '''
        corr=pg.partial_corr(pdData,x='respRDM',y='shapeRDM',x_covar=['numRDM','fsRDM','isRDM'],alternative='two-sided',method='spearman') 
        RDMcorrShape[tp] = corr['r']
        RDMpShape[tp] = corr['p-val']
    '''
    elif method == 'Kendall':
        RDMcorrNum[subj,tp], RDMpNum[subj,tp] = kendalltau(RDMtmp,numRDM)
    
        RDMcorrFs[subj,tp], RDMpFs[subj,tp] = kendalltau(RDMtmp,fsRDM)
         
        RDMcorrIs[subj,tp], RDMpIs[subj,tp] = kendalltau(RDMtmp,isRDM)
            
        RDMcorrShape[subj,tp], RDMpShape[subj,tp] = kendalltau(RDMtmp,shapeRDM)
    '''
'''
elif partial == False:
    corr=pg.corr(x=RDMtmp,y=numRDM,tail='two-sided',method='spearman') 
    RDMcorrNum[subj,tp] = corr['r']
    RDMpNum[subj,tp] = corr['p-val']

    corr=pg.corr(x=RDMtmp,y=fsRDM,tail='two-sided',method='spearman') 
    RDMcorrFs[subj,tp] = corr['r']
    RDMpFs[subj,tp] = corr['p-val']
    
    corr=pg.corr(x=RDMtmp,y=isRDM,tail='two-sided',method='spearman') 
    RDMcorrIs[subj,tp] = corr['r']
    RDMpIs[subj,tp] = corr['p-val']
    
    corr=pg.corr(x=RDMtmp,y=shapeRDM,tail='two-sided',method='spearman') 
    RDMcorrShape[subj,tp] = corr['r']
    RDMpShape[subj,tp] = corr['p-val']
'''        

plt.plot((np.arange(-30,tps-30))/3,RDMcorrNum,label='Number',color='brown')
plt.plot((np.arange(-30,tps-30))/3,RDMcorrFs,label='Field size',color='mediumblue')
plt.plot((np.arange(-30,tps-30))/3,RDMcorrIs,label='Item size',color='forestgreen') #darkorange
plt.plot((np.arange(-30,tps-30))/3,RDMcorrShape,label='Shape',color='black')

# plot the significant line
th = 0.01
thCorr = np.min([np.min(RDMcorrNum),np.min(RDMcorrFs),np.min(RDMcorrIs),np.min(RDMcorrShape)])

#plt.plot(range(-10,tps-10),corrAvgLLF,label='low-level feature',color='black')
# plot the significant line
RDMpNum[(RDMpNum>th)] = None
RDMpNum[(RDMpNum<=th)] = thCorr - 0.01
RDMpFs[(RDMpFs>th)] = None
RDMpFs[(RDMpFs<=th)] = thCorr - 0.02
RDMpIs[(RDMpIs>th)] = None
RDMpIs[(RDMpIs<=th)] = thCorr - 0.03
RDMpShape[(RDMpShape>th)] = None
RDMpShape[(RDMpShape<=th)] = thCorr - 0.04
plt.plot((np.arange(-30,tps-30))/3,RDMpNum,color='brown')
plt.plot((np.arange(-30,tps-30))/3,RDMpFs,color='mediumblue')
plt.plot((np.arange(-30,tps-30))/3,RDMpIs,color='forestgreen')
plt.plot((np.arange(-30,tps-30))/3,RDMpShape,color='black')


plt.xlabel('Time points(10ms)')
plt.ylabel('Partial spearman correlation')
# 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
plt.title('Time course of correlations between MEG RDMs and model RDMs') # partial Spearman
plt.legend()
plt.show()

plotAverage = True
# plot average acc
# we should use raw data here
if plotAverage == True:
    partialAvgAcc = np.average(data, axis=(1, 2, 3))
    partialAvgAcc = np.squeeze(partialAvgAcc)
    x = partialAvgAcc.shape
    plt.plot((np.arange(-30,tps-30))/3,partialAvgAcc)
    plt.xlabel('Time points(10ms)')
    plt.ylabel('Decoding accuracy')
    plt.title('Pairwise decoding accuracy(average)')
    plt.legend()
    plt.show()   







