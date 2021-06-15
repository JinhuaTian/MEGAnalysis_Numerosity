'''
CalRDM:
1. concatenate all data at subject level
2. down-sample data to 100 hz, slice data -0.1~0.7s
3. arrange data: picks=meg, normalize the data(mne.decoding.Scaler)
4. compute MEG RDM: pairwise SVM classification
Run RSA:
5. compute model RDM: number, field size, item size
    arrange label
6. compute the partial Spearman correlation between MEG RDM and model RDM
'''
import mne
import numpy as np
import os
from os.path import join as pj
import matplotlib
from mne.transforms import Transform
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time

matplotlib.use('Qt5Agg') #TkAgg

from numba import jit
jit(nopython=True,parallel=True) #nopython=True,parallel=True

# basic info
rootDir = '/nfs/s2/userhome/tianjinhua/workingdir/meg/numerosity'
subjName = ['subj006']
# 'subj002','subj003','subj004','subj005','subj006','subj007','subj008','subj009','subj010'

scaler = StandardScaler()

newSamplingRate = 100
repeat = 100
kfold = 3
labelNum = 80
tpoints = 80

# compute pair number:
indexNum = 0
for x in range(80):
    for y in range(80):
        if x != y and x + y < 80: # exclude same id
            indexNum = indexNum + 1

accs = np.zeros([len(subjName),tpoints,repeat,kfold,indexNum]) # sub,time,[label-1,label-1],fold,repeat

subIndex = 0

# compute MEG RDM using pairwise SVM classification
for subj in subjName:
    print('subject ' + subj +' is running.')
    savePath = pj(rootDir, subj, 'preprocessed')
    epochs_list = []
    # walk through subj path, concatenate single subject's data to one file
    for file in os.listdir(savePath):
        if 'filterEpochOnly_' in file:
            fifpath = pj(savePath, file)
            epoch = mne.read_epochs(fifpath, preload=True, verbose=True)
            epoch.info['dev_head_t'] = Transform('meg', 'head', np.identity(4))
            epochs_list.append(epoch)
            del epoch
    epochs_all = mne.concatenate_epochs(epochs_list)

    # downsample to 100Hz
    epochs_all.resample(
        sfreq=newSamplingRate,
        npad="auto",
        window="boxcar",
        n_jobs=4,
        pad="edge",
        verbose=True)

    X = epochs_all.get_data(picks = 'meg') # exclude other sensor; # MEG signals: n_epochs, n_meg_channels, n_times
    
    # slice data -0.1~0.7s
    X = X[:,:,10:tpoints+10] # should not del it

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
    pca = PCA(n_components=0.99, svd_solver="full")  # n_components=0.90,
    pca = pca.fit(X)
    Xpc = pca.transform(X)
    # restore the PC number
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
        # normalize the data
        Xt = scaler.fit_transform(Xt)

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
                            acc = svm.score(Pd2[:,1:],Pd2[:,0]) # sub,time,RDMindex,fold,repeat # subIndex,t,re,foldIndex,RDMindex
                            # save acc
                            accs[subIndex,t,re,foldIndex,RDMindex]=acc
                            RDMindex = RDMindex + 1
                foldIndex = foldIndex + 1
        time_elapsed = time.time() - time0
        print('Time point {} finished in {:.0f}m {:.0f}s'.format(t, time_elapsed // 60, time_elapsed % 60)) # + 'repeat '+ str(re)
    subIndex = subIndex + 1

# save MEG RDM
np.save("/nfs/s2/userhome/tianjinhua/workingdir/meg/numerosity/results/MEGRDM3x100_subj6.npy",accs)

partialAvgAcc = np.average(accs, axis=(2, 3, 4))

import matplotlib.pyplot as plt
partialAvgAcc = np.squeeze(partialAvgAcc)
x = partialAvgAcc.shape
plt.plot(range(-10,x[0]-10),partialAvgAcc)
plt.xlabel('Time points(10ms)')
plt.ylabel('Decoding accuracy')
plt.title('Pairwise decoding accuracy(average)')

'''
# --------------------------------------
# 1.2 run GLM per channel per time point
# --------------------------------------
import statsmodels.api as sm

_,m,n = encoderData.shape # m for channel number, n for time points
pVal = []
betaVal = []
for m in range(m):
    for i in range(m):
        gamma_model = sm.GLM(encoderData[], label[i,:], family=sm.families.Gamma())
        gamma_results = gamma_model.fit()

#encoder.plot()
'''
