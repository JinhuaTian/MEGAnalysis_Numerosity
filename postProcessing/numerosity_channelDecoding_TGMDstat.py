# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 20:29:48 2021

@author: tclem
"""
import numpy as np
from os.path import join as pj
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import pandas as pd
import matplotlib.pyplot as plt


def plotMatrix(data, title, vmin=0.45, vmax=0.6):
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
    plt.ylabel('Training time')
    plt.xlabel('Testing time')
    plt.title(title)
    plt.show()

rootDir = 'E:\temp2\TG_MD'
# make stimulus RDM
eventMatrix = np.loadtxt('E:/temp2/STI.txt')
# '004','005','006','007','009','011','012','013','014','015','016','017','018','020','021','022','023'
subjs = ['004','005','006','007','009','011','012','013','014','015','016','017','018','020','021','022','023'] # #'4','5','6',
timepoints = 240

# make correlation matrix

numRDMs = np.zeros((len(subjs),timepoints,timepoints))
fsRDMs = np.zeros((len(subjs),timepoints,timepoints))
isRDMs = np.zeros((len(subjs),timepoints,timepoints))
shapeRDMs = np.zeros((len(subjs),timepoints,timepoints))


subjIndex = 0
subjNum = 0
for subj in subjs:
    fileName = 'MDTG15x300hz_subj'+ subj + '.npy'
    filePath = pj(rootDir, fileName)
    data = np.load(filePath) # subIndex,t,re,foldIndex,RDMindex
    
    numRDMs[subjIndex,:,:] = data[0,:,:]
    fsRDMs[subjIndex,:,:] = data[1,:,:]
    isRDMs[subjIndex,:,:] = data[2,:,:]
    shapeRDMs[subjIndex,:,:] = data[3,:,:]
    
    subjIndex = subjIndex+1
    del data


avgnumRDM = np.average(numRDMs,axis=0)
avgfsRDM = np.average(fsRDMs,axis=0)
avgisRDM = np.average(isRDMs,axis=0)
avgshapeRDM = np.average(shapeRDMs,axis=0)

plotMatrix(avgnumRDM,'Temporal generalization matrix (number)',vmax=0.6)
plotMatrix(avgfsRDM,'Temporal generalization matrix (field size)',vmax=0.6)
plotMatrix(avgisRDM,'Temporal generalization matrix (item size)',vmax=0.6)
plotMatrix(avgshapeRDM,'Temporal generalization matrix (shape)',vmax=0.6)

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
cax = ax.matshow(RDMs[i], cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1,
fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
plt.title(plotPics[i] +' RDM')
plt.show()
'''