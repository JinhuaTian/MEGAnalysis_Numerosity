# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 22:57:45 2021

@author: tclem
"""
from os.path import join as pj
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr

eventMatrix =  np.loadtxt('C:/Users/tclem/Documents/GitHub/MEGAnalysis_Numerosity/postProcessing/STI.txt')

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
            
# calculate low-level feature
stiPath = r'E:\meg\DATA-IBP\mrdm\stim'
imgs = []
for i in range(1,81):
    imgs.append(str(i)+'.png')
 
# make empty image matrix
img = Image.open(pj(stiPath,imgs[0]))
img = np.array(img)

a,b,c=img.shape
imgNum = 80   

imgMatrix = np.zeros((imgNum,a*b*c))
TFAMatrix = np.zeros((imgNum,a))

num = 0
for img in imgs:
    imgpath = pj(stiPath,img)
    imgfile = Image.open(imgpath)
    imgfile = np.array(imgfile)
    imgfile = imgfile.reshape(a*b*c)
    imgMatrix[num,:]=imgfile
    TFAMatrix[num,:]=sum(imgfile==0)+sum(imgfile==255)
    num = num + 1

# normarlize data
scaler = StandardScaler()
imgMatrix = scaler.fit_transform(imgMatrix)

LLFRDM = np.zeros((imgNum,imgNum))
TFARDM = np.zeros((imgNum,imgNum))

corrArray = []
TFAArray = []

# RDM, use 1-r instead
for x in range(labelNum):
    for y in range(x+1,labelNum):
        r = pearsonr(imgMatrix[x,:],imgMatrix[y,:])[0]
        corrArray.append(1-r)
        LLFRDM[x,y] = 1-r
        TFAArray.append(abs(TFAMatrix[x,:]-TFAMatrix[y,:]))
        TFARDM[x,y] = abs(TFAMatrix[x,:]-TFAMatrix[y,:])

#nomarlize to [0,1]
def normalization(data):
    data = np.array(data)
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
'''
def flip90_left(arr):
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr
'''
LLFRDM = normalization(LLFRDM) # full RDM
TFARDM = normalization(TFARDM)
# plot LLF RDM
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 6), dpi=300) #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(LLFRDM, cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1  ,
fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
plt.title('Low-level feature RDM')
plt.show()

# plot TFA RDM
fig = plt.figure(figsize=(8, 6), dpi=300) #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(TFARDM, cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1  ,
fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
plt.title('Low-level feature RDM')
plt.show()

corrArray = normalization(corrArray) # LLF RDM vector, half of the RDM
TFAArray = normalization(TFAArray)
# np.save('C:/Users/tclem/Desktop/MEG/LowLevelMatrix.npy',corrArray)
























