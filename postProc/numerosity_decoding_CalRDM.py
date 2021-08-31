# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:22:51 2021

@author: tclem
"""
'''
CalRDM:
1. concatenate all data at subject level
2. down-sample data to 300 hz, slice data -0.1~0.7s
3. arrange data: picks=meg, normalize the data(mne.decoding.Scaler)
4. compute MEG RDM: pairwise SVM classification
Run RSA:
5. compute model RDM: number, field size, item size
    arrange label
6. compute the partial Spearman correlation between MEG RDM and model RDM
'''
import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time
import matplotlib
matplotlib.use('Qt5Agg') #TkAgg
#from libsvm import svmutil as sv # pip install -U libsvm-official
#import sys
#sys.path.append('/nfs/a2/userhome/tianjinhua/workingdir/meg/mne/')
import mne
from mne.transforms import Transform
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from numba import jit
jit(nopython=True,parallel=True) #nopython=True,parallel=True

# basic info
rootDir = 'D:/temp'
#subjName = ['subj016']
subjid = 'subj004'
# 'subj002','subj003','subj004','subj005','subj006','subj007','subj008','subj009','subj010'

newSamplingRate = 300
repeat = 100
kfold = 3
labelNum = 80
tpoints = 240

# compute pair number:
indexNum = 0
for x in range(80):
    for y in range(80):
        if x != y and x + y < 80: # exclude same id
            indexNum = indexNum + 1

accs = np.zeros([tpoints,repeat,kfold,indexNum]) # time,[label-1,label-1],fold,repeat

pcNum = [] # restore the number of PCs

# compute MEG RDM using pairwise SVM classification

print('subject ' + subjid +' is running.')
savePath = pj(rootDir, subjid, 'preprocessed')
epochs_list = []
# walk through subj path, concatenate single subject's data to one file
for file in os.listdir(savePath):
    if 'filterEpochICAMR_' in file:
        fifpath = pj(savePath, file)
        epoch = mne.read_epochs(fifpath, preload=False, verbose=True)
        epoch.info['dev_head_t'] = Transform('meg', 'head', np.identity(4)) # ctf use mag instead of meg
        epochs_list.append(epoch)
        del epoch
epochs_all = mne.concatenate_epochs(epochs_list)

# downsample to 300Hz
epochs_all.resample(
    sfreq=newSamplingRate,
    npad="auto",
    window="boxcar",
    # n_jobs=4,
    pad="edge",
    verbose=True)

X = epochs_all.get_data(picks = 'mag') # exclude other sensor; # MEG signals: n_epochs, n_meg_channels, n_times
print("X shape is",str(X.shape))
# slice data -0.1~0.7s
X = X[:,:,30:tpoints+30] # should not del it

nEpochs, nChan, nTime = X.shape
# training label
Y = epochs_all.events[:, 2]
Y = Y.reshape(nEpochs,1)
del epochs_all
# make new label
Yshape = Y.shape[0]
	
# Apply PCA to reduce dimension. It works bad, just reduce 306 dim to 276 dim
# perform pca according to the variance remains
# pcNum = []  # restore the number of PCs
X = np.transpose(X,[0,2,1]) #change (n_epochs, n_meg_channels, n_times) to (nEpochs * nTime, nChan)
X = X.reshape(nEpochs*nTime, nChan)
X = scaler.fit_transform(X) #normarlize data
pca = PCA(n_components=0.99, svd_solver="full")  # n_components=0.90,
pca = pca.fit(X)
Xpc = pca.transform(X)

# normalize PC
Xpc = scaler.fit_transform(Xpc)

# print the PC number
# pcNum.append(Xpc.shape[1])
print('PC number is '+ str(Xpc.shape[1]))
#nEpochs, nChan, nTime = X.shape
X = Xpc.reshape(nEpochs,nTime,Xpc.shape[1])
del Xpc
X = np.transpose(X,[0,2,1])

# RDM of decoding accuracy values for each time point
for t in range(nTime):
    # pick the time data and normalize data
    Xt = X[:,:,t]
    # Xt = scaler.fit_transform(Xt)

    time0 = time.time()
    #repeat for repeat times:
    for re in range(repeat):
        state = np.random.randint(0,100)
        kf=StratifiedKFold(n_splits=kfold, shuffle=True,random_state=state)
        foldIndex = 0
        for train_index, test_index in kf.split(Xt,Y):
            xTrain, xTest, yTrain, yTest, = Xt[train_index], Xt[test_index],Y[train_index],Y[test_index]
            trainPd = np.concatenate((yTrain,xTrain),axis=1) # train data
            testPd = np.concatenate((yTest,xTest),axis=1) # test data
            RDMindex = 0
            for x in range(80):
                for y in range(80):
                    if x != y and x + y < 80: #x + y < 82:
                        Pd1 = trainPd[(trainPd[:,0] == (x+1)) | (trainPd[:,0] == (y+1))] # labels are 1~80
                        Pd2 = testPd[(testPd[:,0] == (x+1)) | (testPd[:,0] == (y+1))]
                        # run svm
                        svm = SVC(kernel="linear")
                        svm.fit(Pd1[:,1:],Pd1[:,0])
                        acc = svm.score(Pd2[:,1:],Pd2[:,0]) # sub,time,RDMindex,fold,repeat # t,re,foldIndex,RDMindex
                        # save acc
                        accs[t, re, foldIndex, RDMindex] = acc
                        '''
                        model = sv.svm_train(Pd1[:,0],Pd1[:,1:], "-q -t 0")
                        _, p_acc, _ = sv.svm_predict(Pd2[:,0],Pd2[:,1:], model, "-q")#p_label, p_acc, p_val = 
                        # save acc
                        accs[t,re,foldIndex,RDMindex]=p_acc[0]
                        '''
                        RDMindex = RDMindex + 1
            foldIndex = foldIndex + 1
    time_elapsed = time.time() - time0
    print('Time point {} finished in {:.0f}m {:.0f}s'.format(t, time_elapsed // 60, time_elapsed % 60)) # + 'repeat '+ str(re)


# save MEG RDM
fileName = pj(rootDir, 'ctfRDM3x100x300hz_'+ subjid +'.npy')
np.save(fileName,accs)

'''
partialAvgAcc = np.average(accs, axis=(2, 3, 4))

import matplotlib.pyplot as plt
partialAvgAcc = np.squeeze(partialAvgAcc)
x = partialAvgAcc.shape
plt.plot(range(-30,x[0]-30),partialAvgAcc)
plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.title('Pairwise decoding accuracy(average)')
'''