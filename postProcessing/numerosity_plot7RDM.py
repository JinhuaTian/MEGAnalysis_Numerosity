# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:31:51 2022

@author: tclem
"""
import numpy as np
from pandas.core.frame import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
data = np.load('E:/temp2/ModelRDM_NumFsIsShapeTfaDenLLF.npy')
data = DataFrame(data)
corr_all = data.T.corr(method='spearman')

fig = plt.figure(figsize=(8,8), dpi=300) #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(corr_all, cmap='jet',vmin=0, vmax=1)  #绘制热力图，从-1到1  ,
plt.title("RDMs' Spearman correlation")
# fig.colorbar(cax)  #cax将matshow生成热力图设置为颜色渐变条
cbar = fig.colorbar(cax, ticks=[0, 1])
#ax.spines['bottom'].set_xticklabels(['','Number','Field area','Item area','Shape', 'Total Field area', 'Density','Low-level feature'],fontdict={'size': 10, 'color': 'black'})
ax.set_xticklabels(['','Number','Field area','Item area','Shape', 'Total Field area', 'Density','Low-level feature'],fontdict={'size': 10, 'color': 'black'})
ax.set_yticklabels(['','Number','Field area','Item area','Shape', 'Total Field area', 'Density','Low-level feature'],fontdict={'size': 10, 'color': 'black'})
#plt.tight_layout()
plt.savefig('C:/Users/tclem/Desktop/image/RDM.png')
plt.show()
