# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 22:57:45 2021

@author: tclem

calculate Number, field size, item size, shape, density (field size/number), TFA (item size x number) 
"""
from os.path import join as pj
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#import torchvision.transforms as transforms
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

eventMatrix =  np.loadtxt('C:/Users/tclem/Documents/GitHub/MEGAnalysis_Numerosity/postProcessing/STI.txt')

# make correlation matrix
index = 0
numRDM = []
fsRDM = []
isRDM = []
shapeRDM = []
denRDM = []
tfaRDM = []
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
        denRDM.append(abs(eventMatrix[x,1]/eventMatrix[x,0]-eventMatrix[y,1]/eventMatrix[y,0]))
        tfaRDM.append(abs(eventMatrix[x,2]*eventMatrix[x,0]-eventMatrix[y,2]*eventMatrix[y,0]))
            
# calculate low-level feature
stiPath = r'E:\meg\DATA-IBP\mrdm\stim'
imgs = []
for i in range(1,81):
    imgs.append(str(i)+'.png')

# make empty image matrix
img = Image.open(pj(stiPath,imgs[0]))
img = img.convert('1') # should not use "L"
# img.show()
img = np.array(img)

a,b=img.shape
imgNum = 80   

imgMatrix = np.zeros((imgNum,a*b))

num = 0
for img in imgs:
    imgpath = pj(stiPath,img)
    imgfile = Image.open(imgpath)
    # convert to gray scale
    imgfile = imgfile.convert('L')
    imgfile = np.array(imgfile)
    imgfile = imgfile.reshape(a*b)
    imgMatrix[num,:]=imgfile
    num = num + 1

# normarlize data
scaler = StandardScaler()
imgMatrix = scaler.fit_transform(imgMatrix)

LLFRDM = np.zeros((imgNum,imgNum))
corrArray = []
TFAArray = []

# RDM, use 1-r instead
for x in range(labelNum):
    for y in range(x+1,labelNum):
        r = pearsonr(imgMatrix[x,:],imgMatrix[y,:])[0]
        corrArray.append(1-r)
        LLFRDM[x,y] = 1-r

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

# normalize RDM
# LLFRDM = normalization(LLFRDM) # no need to normalize data
# nomalize RDM vector
corrArray = normalization(corrArray) # LLF RDM vector, half of the RDM
denRDM = normalization(denRDM)
tfaRDM = normalization(tfaRDM)


# plot LLF RDM
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 6), dpi=300) #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(LLFRDM, cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1  ,
fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
plt.title('Low-level feature RDM')
plt.show()
'''
# plot TFA RDM
fig = plt.figure(figsize=(6, 6), dpi=300) #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(TFARDM, cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1  ,
fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
plt.title('Low-level feature RDM')
plt.show()
'''

# np.save('C:/Users/tclem/Desktop/MEG/LowLevelMatrix.npy',corrArray)

modelRDM = np.array([numRDM,fsRDM,isRDM,shapeRDM,tfaRDM,denRDM,corrArray])
np.save('E:/temp2/ModelRDM_NumFsIsShapeTfaDenLLF.npy',modelRDM) 





















