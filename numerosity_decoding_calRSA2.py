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

filePath = r'C:\Users\tclem\Desktop\MEG\MEGRDM3x100_subj7.npy'

# make stimulus RDM
eventMatrix = np.array([[1,1,1,1],[1,1,1,2],[1,1,1,3],[1,1,1,4],[1,1,1,5],[1,1,1,6],[1,1,1,7],[1,1,1,8],[1,1,1,9],[1,1,1,10],
                [1,1,2,11],[1,1,2,12],[1,1,2,13],[1,1,2,14],[1,1,2,15],[1,1,2,16],[1,1,2,17],[1,1,2,18],[1,1,2,19],[1,1,2,20],
                [1,2,1,21],[1,2,1,22],[1,2,1,23],[1,2,1,24],[1,2,1,25],[1,2,1,26],[1,2,1,27],[1,2,1,28],[1,2,1,29],[1,2,1,30],
                [1,2,2,31],[1,2,2,32],[1,2,2,33],[1,2,2,34],[1,2,2,35],[1,2,2,36],[1,2,2,37],[1,2,2,38],[1,2,2,39],[1,2,2,40],
                [2,1,1,41],[2,1,1,42],[2,1,1,43],[2,1,1,44],[2,1,1,45],[2,1,1,46],[2,1,1,47],[2,1,1,48],[2,1,1,49],[2,1,1,50],
                [2,1,2,51],[2,1,2,52],[2,1,2,53],[2,1,2,54],[2,1,2,55],[2,1,2,56],[2,1,2,57],[2,1,2,58],[2,1,2,59],[2,1,2,60],
                [2,2,1,61],[2,2,1,62],[2,2,1,63],[2,2,1,64],[2,2,1,65],[2,2,1,66],[2,2,1,67],[2,2,1,68],[2,2,1,69],[2,2,1,70],
                [2,2,2,71],[2,2,2,72],[2,2,2,73],[2,2,2,74],[2,2,2,75],[2,2,2,76],[2,2,2,77],[2,2,2,78],[2,2,2,79],[2,2,2,80]])


pixel = [1.72939142e-03, 8.23519723e-04, 2.14115128e-03, 1.23527958e-03,
       1.39998353e-03, 1.07057564e-03, 1.31763156e-03, 2.47055917e-03,
       2.22350325e-03, 2.71761509e-03, 4.80523759e-01, 4.78464959e-01,
       4.77723791e-01, 4.79782591e-01, 4.80029647e-01, 4.78547311e-01,
       4.81347278e-01, 4.77476736e-01, 4.78712015e-01, 4.78217903e-01,
       1.64703945e-03, 2.63526311e-03, 3.21172692e-03, 5.76463806e-04,
       2.96467100e-03, 1.07057564e-03, 9.88223668e-04, 3.87054270e-03,
       8.23519723e-05, 1.56468747e-03, 4.77723791e-01, 4.78876719e-01,
       4.81841390e-01, 4.79206127e-01, 4.76570864e-01, 4.79947295e-01,
       4.79206127e-01, 4.77064976e-01, 4.76241456e-01, 4.79453183e-01,
       1.57374619e-01, 1.61656922e-01, 1.60668698e-01, 1.61739274e-01,
       1.62233385e-01, 1.59186363e-01, 1.58115787e-01, 1.59762826e-01,
       1.58527547e-01, 1.59845178e-01, 9.98517664e-01, 9.97282385e-01,
       9.94894178e-01, 9.93823602e-01, 9.99176480e-01, 9.98517664e-01,
       9.94729474e-01, 9.97694145e-01, 9.94729474e-01, 9.94070658e-01,
       1.57292267e-01, 1.60833402e-01, 1.58445195e-01, 1.59351066e-01,
       1.58527547e-01, 1.59598122e-01, 1.58527547e-01, 1.60998106e-01,
       1.58362843e-01, 1.57374619e-01, 9.95058882e-01, 1.00008235e+00,
       9.94482418e-01, 9.88553076e-01, 9.93823602e-01, 9.94070658e-01,
       9.91105987e-01, 9.94729474e-01, 9.91517747e-01, 9.93658898e-01]

# make correlation matrix
index = 0
numRDM = []
fsRDM = []
isRDM = []
LLFRDM = np.load('C:/Users/tclem/Desktop/MEG/LowLevelMatrix.npy') # low-level features
IARDM = []

for x in range(80):
    for y in range(80):
        if x != y and x + y < 80: #x + y < 80:
            # num RDM
            if eventMatrix[x,0] == eventMatrix[y,0]:
                numRDM.append(0)
            else:
                numRDM.append(1)
                
            '''
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
            '''
            # item area RDM
            IARDM.append(pixel[x]/pixel[y])
            


# compute partial spearman correlation, with other 2 RDM controlled 
data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex 
subIndex,t,re,foldIndex,RDMindex = data.shape
data = data.reshape(subIndex*t*re*foldIndex,RDMindex)
# normalize the MEG RDM to [0,1]
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(data)
data = data.reshape(subIndex,t,re,foldIndex,RDMindex)

# make 3 x 2 empty matrix 
subjs, tps, RDM, fold, repeats = data.shape
RDMcorrNum = np.zeros((subjs,tps))
RDMpNum = np.zeros((subjs,tps))

RDMcorrIA = np.zeros((subjs,tps))
RDMpIA = np.zeros((subjs,tps))
RDMcorrLLF = np.zeros((subjs,tps))
RDMpLLF = np.zeros((subjs,tps))

for subj in range(subjs):
    for tp in range(tps):
        datatmp = data[subj, tp,:] # subIndex,t,re,foldIndex,RDMindex 
        RDMtmp = np.average(data[subj,tp,:,:,:], axis=(0, 1))
        
        pdData = pd.DataFrame({'respRDM':RDMtmp,'numRDM':numRDM,'IARDM':IARDM,'LLFRDM':LLFRDM})
        
        corr=pg.partial_corr(pdData,x='respRDM',y='numRDM',x_covar=['IARDM','LLFRDM'],tail='two-sided',method='spearman') 
        RDMcorrNum[subj,tp] = corr['r']
        RDMpNum[subj,tp] = corr['p-val']
    
        corr=pg.partial_corr(pdData,x='respRDM',y='IARDM',x_covar=['numRDM','LLFRDM'],tail='two-sided',method='spearman') 
        RDMcorrIA[subj,tp] = corr['r']
        RDMpIA[subj,tp] = corr['p-val']
        
        corr=pg.partial_corr(pdData,x='respRDM',y='LLFRDM',x_covar=['numRDM','IARDM'],tail='two-sided',method='spearman') 
        RDMcorrLLF[subj,tp] = corr['r']
        RDMpLLF[subj,tp] = corr['p-val']
        
# average cross different 
corrAvgNum = np.average(RDMcorrNum,axis=0)
pAvgNum = np.average(RDMpNum,axis=0)
corrAvgIA = np.average(RDMcorrIA,axis=0)
pAvgIA = np.average(RDMpIA,axis=0)
corrAvgLLF= np.average(RDMcorrLLF,axis=0)
pAvgLLF = np.average(RDMpLLF,axis=0)

import matplotlib.pyplot as plt
plt.plot(range(-10,tps-10),corrAvgNum,label='Number',color='brown')
plt.plot(range(-10,tps-10),corrAvgIA,label='item area',color='mediumblue')
plt.plot(range(-10,tps-10),corrAvgLLF,label='low-level feature',color='forestgreen') #darkorange
# plot the significant line
pAvgNum[(pAvgNum>0.05)] = None
pAvgNum[(pAvgNum<=0.05)] = -0.15
pAvgIA[(pAvgIA>0.05)] = None
pAvgIA[(pAvgIA<=0.05)] = -0.17
pAvgLLF[(pAvgLLF>0.05)] = None
pAvgLLF[(pAvgLLF<=0.05)] = -0.21
plt.plot(range(-10,tps-10),pAvgNum,color='brown')
plt.plot(range(-10,tps-10),pAvgIA,color='mediumblue')
plt.plot(range(-10,tps-10),pAvgLLF,color='forestgreen')

plt.xlabel('Time points(10ms)')
plt.ylabel('Partial spearman correlation')
# 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
plt.title('Time course of partial Spearman correlations between MEG RDMs and model RDMs')
plt.legend(loc='best')
plt.show()

# plot average acc
partialAvgAcc = np.average(data, axis=(2, 3, 4))

import matplotlib.pyplot as plt
partialAvgAcc = np.squeeze(partialAvgAcc)
x = partialAvgAcc.shape
plt.plot(range(-10,x[0]-10),partialAvgAcc)
plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.title('Pairwise decoding accuracy(average)')
plt.show()