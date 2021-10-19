# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:20:04 2021

@author: tclem
"""

import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from pandas.core.frame import DataFrame
import pandas as pd
import matplotlib.pyplot as plt

# basic info
rootDir = 'E:/temp2'
subjid = 'subj019' #subjName = ['subj016']
fileName = pj(rootDir, 'ctfTG3x100x300hz_'+ subjid +'.npy')
tgData = np.load(fileName)

avgData = np.average(tgData,axis=2)

def plotMatrix(data, title, vmin=0.4, vmax=0.7):
    data = pd.DataFrame(data,dtype='float64')
    fig = plt.figure(figsize=(8, 6), dpi=100) #调用figure创建一个绘图对象
    cax = fig.add_subplot(111)
    cax = cax.matshow(data, vmin=vmin, vmax=vmax)  # cmap='jet',绘制热力图，从-1到1  ,
    ax = plt.gca()
    #ax.invert_xaxis()
    ax.invert_yaxis()
    fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
    #ax.set_xticklabels(['number','field size','item size','low-level feature'],fontdict={'size': 10, 'color': 'black'})
    #ax.set_yticklabels(['number','field size','item size','low-level feature'],fontdict={'size': 10, 'color': 'black'})
    # plt.xlim(np.linspace(-100, 100, 700))

    plt.title(title)
    plt.show()

plotMatrix(avgData,'Temporal generalization matrix',vmax=0.65)

# make stimulus RDM
eventMatrix = np.loadtxt('E:/temp2/STI.txt')
# '004','005','006','007','009','011','012','013','014','015','016','017','018','020','021','022','023'
subjs = ['019','020','021','022','023'] # #'4','5','6',
timepoints = 240

# make correlation matrix
subjIndex = 0
numRDMs = np.zeros((len(subjs),timepoints,timepoints))
fsRDMs = np.zeros((len(subjs),timepoints,timepoints))
isRDMs = np.zeros((len(subjs),timepoints,timepoints))
shapeRDMs = np.zeros((len(subjs),timepoints,timepoints))

labelNum = 80

subjNum = 0
for subj in subjs:
    fileName = 'ctfTG3x100x300hz_subj'+ subj + '.npy'
    filePath = pj(rootDir, fileName)
    
    data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex

    index = 0
    numRDM = []
    fsRDM = []
    isRDM = []
    shapeRDM = []

    for x in range(labelNum):
        for y in range(x+1,labelNum):
            # num RDM
            if eventMatrix[x,1] == eventMatrix[y,1] and eventMatrix[x,2] == eventMatrix[y,2] and eventMatrix[x,3] == eventMatrix[y,3]:
                numRDM.append(data[:,:,index])
                index = index + 1
            elif eventMatrix[x,0] == eventMatrix[y,0] and eventMatrix[x,2] == eventMatrix[y,2] and eventMatrix[x,3] == eventMatrix[y,3]:
                fsRDM.append(data[:,:,index])
                index = index + 1
            elif eventMatrix[x,0] == eventMatrix[y,0] and eventMatrix[x,1] == eventMatrix[y,1] and eventMatrix[x,3] == eventMatrix[y,3]:
                isRDM.append(data[:,:,index])
                index = index + 1
            elif eventMatrix[x,0] == eventMatrix[y,0] and eventMatrix[x,1] == eventMatrix[y,1] and eventMatrix[x,2] == eventMatrix[y,2]:
                shapeRDM.append(data[:,:,index])
                index = index + 1
            else:
                index = index + 1
    
    numRDM = np.array(numRDM).T
    numRDM = np.average(numRDM,axis=2)
    numRDMs[subjIndex,:] = numRDM
    fsRDM = np.array(fsRDM).T
    fsRDM = np.average(fsRDM,axis=2)
    fsRDMs[subjIndex,:] = fsRDM
    isRDM = np.array(isRDM).T
    isRDM = np.average(isRDM,axis=2)
    isRDMs[subjIndex,:] = isRDM
    shapeRDM = np.array(shapeRDM).T
    shapeRDM = np.average(shapeRDM,axis=2)
    shapeRDMs[subjIndex,:] = shapeRDM
    
    subjIndex = subjIndex+1
    del data

avgnumRDM = np.average(numRDMs,axis=0)
avgfsRDM = np.average(fsRDMs,axis=0)
avgisRDM = np.average(isRDMs,axis=0)
avgshapeRDM = np.average(shapeRDMs,axis=0)

plotMatrix(avgnumRDM,'Temporal generalization matrix (number)',vmax=0.65)


plotMatrix(avgfsRDM,'Temporal generalization matrix (field size)',vmax=0.65)
plotMatrix(avgisRDM,'Temporal generalization matrix (item size)',vmax=0.65)
plotMatrix(avgshapeRDM,'Temporal generalization matrix (shape)',vmax=0.65)

'''
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
'''

'''
fig = plt.figure(figsize=(8, 6), dpi=100) #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(RDMs[i], cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1  ,
fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
plt.title(plotPics[i] +' RDM')
plt.show()
'''