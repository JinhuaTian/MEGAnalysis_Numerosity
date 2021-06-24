# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:57:15 2021

@author: tclem
"""
import numpy as np
#import pingouin as pg
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

subjs = ['2','3','7','8','9','10'] # #'4','5','6',
path =  r'C:\Users\tclem\Desktop\MEG'
timepoints = 80

# make correlation matrix
subjIndex = 0
numRDMs = np.zeros((len(subjs),timepoints))
fsRDMs = np.zeros((len(subjs),timepoints))
isRDMs = np.zeros((len(subjs),timepoints))

subjNum = 0
for subj in subjs:
    fileName = 'MEGRDM3x100_subj'+ subj + '.npy'
    filePath = pj(path, fileName)
    
    data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex
    dataAcc = np.average(data,axis = (0,2,3)) # 80 time point x 3200 RDM

    index = 0
    numRDM = []
    fsRDM = []
    isRDM = []

    for x in range(80):
        for y in range(80):
            if x != y and x + y < 80: #x + y < 80:
                # num RDM
                if eventMatrix[x,1] == eventMatrix[y,1] and eventMatrix[x,2] == eventMatrix[y,2]:
                    numRDM.append(dataAcc[:,index])
                    index = index + 1
                elif eventMatrix[x,0] == eventMatrix[y,0] and eventMatrix[x,2] == eventMatrix[y,2]:
                    fsRDM.append(dataAcc[:,index])
                    index = index + 1
                elif eventMatrix[x,0] == eventMatrix[y,0] and eventMatrix[x,1] == eventMatrix[y,1]:
                    isRDM.append(dataAcc[:,index])
                    index = index + 1
                else:
                    index = index + 1
    
    numRDM = np.array(numRDM).T
    numRDM = np.average(numRDM,axis=1)
    numRDMs[subjIndex,:] = numRDM
    fsRDM = np.array(fsRDM).T
    fsRDM = np.average(fsRDM,axis=1)
    fsRDMs[subjIndex,:] = fsRDM
    isRDM = np.array(isRDM).T
    isRDM = np.average(isRDM,axis=1)
    isRDMs[subjIndex,:] = isRDM
    
    subjIndex = subjIndex+1
    del data

avgnumRDM = np.average(numRDMs,axis=0)
avgfsRDM = np.average(fsRDMs,axis=0)
avgisRDM = np.average(isRDMs,axis=0)

import matplotlib.pyplot as plt
x = numRDM.shape
plt.plot(range(-10,x[0]-10),avgnumRDM,label='number')
plt.plot(range(-10,x[0]-10),avgfsRDM,label='field size')
plt.plot(range(-10,x[0]-10),avgisRDM,label='item size')
plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.legend()
plt.title('Balanced decoding accuracy(average)')

