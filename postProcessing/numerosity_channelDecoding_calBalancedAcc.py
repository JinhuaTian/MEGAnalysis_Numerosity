# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:57:15 2021

@author: tclem
"""
import numpy as np
#import pingouin as pg
from os.path import join as pj

# make stimulus RDM
eventMatrix = np.loadtxt(r'C:\Users\Clemens\Documents\GitHub\MEGAnalysis_Numerosity\postProcessing\STI.txt')
# '004','005','006','007','009','011','012','013','014','015','016','017','018','020','021','022','023'
subjs = ['004','005','006','007','009','011','012','013','014','015','016','017','018','019','020','021','022','023'] # #'4','5','6',
path =  r'D:\MEG\channelRDM'
timepoints = 240
tps = 240

# make correlation matrix
subjIndex = 0
numRDMs = np.zeros((len(subjs),timepoints))
fsRDMs = np.zeros((len(subjs),timepoints))
isRDMs = np.zeros((len(subjs),timepoints))
shapeRDMs = np.zeros((len(subjs),timepoints))

labelNum = 80

subjNum = 0
for subj in subjs:
    fileName = 'ctfRDM3x100x300hz_subj'+ subj + '.npy'
    filePath = pj(path, fileName)
    
    data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex
    dataAcc = np.average(data,axis = (1,2)) # 80 time point x 3200 RDM

    index = 0
    numRDM = []
    fsRDM = []
    isRDM = []
    shapeRDM = []

    for x in range(labelNum):
        for y in range(x+1,labelNum):
            # num RDM
            if eventMatrix[x,1] == eventMatrix[y,1] and eventMatrix[x,2] == eventMatrix[y,2] and eventMatrix[x,3] == eventMatrix[y,3]:
                numRDM.append(dataAcc[:,index])
                index = index + 1
            elif eventMatrix[x,0] == eventMatrix[y,0] and eventMatrix[x,2] == eventMatrix[y,2] and eventMatrix[x,3] == eventMatrix[y,3]:
                fsRDM.append(dataAcc[:,index])
                index = index + 1
            elif eventMatrix[x,0] == eventMatrix[y,0] and eventMatrix[x,1] == eventMatrix[y,1] and eventMatrix[x,3] == eventMatrix[y,3]:
                isRDM.append(dataAcc[:,index])
                index = index + 1
            elif eventMatrix[x,0] == eventMatrix[y,0] and eventMatrix[x,1] == eventMatrix[y,1] and eventMatrix[x,2] == eventMatrix[y,2]:
                shapeRDM.append(dataAcc[:,index])
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
    
    shapeRDM = np.array(shapeRDM).T
    shapeRDM = np.average(shapeRDM,axis=1)
    shapeRDMs[subjIndex,:] = shapeRDM
    
    subjIndex = subjIndex+1
    del data

avgnumRDM = np.average(numRDMs,axis=0)
avgfsRDM = np.average(fsRDMs,axis=0)
avgisRDM = np.average(isRDMs,axis=0)
avgshapeRDM = np.average(shapeRDMs,axis=0)

import matplotlib.pyplot as plt
x = numRDM.shape
plt.plot((np.arange(-30,tps-30))/3,avgnumRDM,label='Number',color='brown')
plt.plot((np.arange(-30,tps-30))/3,avgfsRDM,label='Field size',color='mediumblue')
plt.plot((np.arange(-30,tps-30))/3,avgisRDM,label='Item size',color='forestgreen')
plt.plot((np.arange(-30,tps-30))/3,avgshapeRDM,label='Shape',color='black')

plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.legend()
plt.title('Balanced decoding accuracy(average)')

