# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 21:01:52 2021

@author: tclem
"""
'''
evoke result procedure:
1.load data
2.filter:notch, high pass, low pass
3.epoch(auto rejection, baseline correction), seg evoked data for each condition


4.inspect epochs
5.average epochs
6.source reconstruction
7.decoding
'''
import mne
import numpy as np
import os
from os.path import join as pj
import matplotlib
# matplotlib.use('TkAgg')

from mne.preprocessing import annotate_movement, compute_average_dev_head_t

#filepath = '/nfs/s2/userhome/tianjinhua/workingdir/meg/numerosity/gu_wenyu/preprocessed/filterEpochOnly_run8_tsss.fif'     run1_tsss.fif
rootDir = 'E:/temp'
subjName = ['subj004'] #,'subj002','subj004','subj005','subj005','subj006'
# 'zhai_yunze','yang_yuqing','zhang_yuhang','subj002','subj003','subj004','subj005','subj006','subj007'
#,'subj1_chenrui','subj2_xuwei','gu_wenyu','sun_baojia','wang_yan'
# 'ding_zheyi',  'zhai_yunze'(loss run)
taskName = 'raw'

#2. concatenate evoked data and visualize results
# define condition
for i in range(1, 17):
    exec('num{} = np.arange(((i-1)*5+1),((i-1)*5+6))'.format(i))

condition = ['evoked_fsS','evoked_fsL','evoked_isS','evoked_isL','evoked_numS','evoked_numL','evoked_cir','evoked_tri']

#fileNum = 15 # run number
conditions = ['CirSnumSfsSis','CirSnumSfsLis','CirSnumLfsSis','CirSnumLfsLis','CirLnumSfsSis','CirLnumSfsLis','CirLnumLfsSis','CirLnumLfsLis',
              'TriSnumSfsSis','TriSnumSfsLis','TriSnumLfsSis','TriSnumLfsLis','TriLnumSfsSis','TriLnumSfsLis','TriLnumLfsSis','TriLnumLfsLis']

for subj in subjName:
    rawDir = pj(rootDir, subj, taskName)
    savePath = pj(rootDir, subj, 'preprocessed')
    fileNum = 0
    i = 0
    for file in os.listdir(savePath):
        if 'filterEpochICAMR_' in file:
            #rawName = raw + str(fileNum)
            filepath = pj(savePath,file)
            #exec('raw{}= mne.io.read_raw_fif(filepath,allow_maxshield=True,preload=True)'.format(fileNum)) #Do we need Signal Space Projection (SSP)?
            epoch = mne.read_epochs(filepath, preload=True) #allow_maxshield=True,

            # compute evoked data for each condition
            exec('evoked_CirSnumSfsSis{} = epoch[num1].average()'.format(i+1))
            exec('evoked_CirSnumSfsLis{} = epoch[num2].average()'.format(i+1))
            exec('evoked_CirSnumLfsSis{} = epoch[num3].average()'.format(i+1))
            exec('evoked_CirSnumLfsLis{} = epoch[num4].average()'.format(i+1))
            exec('evoked_CirLnumSfsSis{} = epoch[num5].average()'.format(i+1))
            exec('evoked_CirLnumSfsLis{} = epoch[num6].average()'.format(i+1))
            exec('evoked_CirLnumLfsSis{} = epoch[num7].average()'.format(i+1))
            exec('evoked_CirLnumLfsLis{} = epoch[num8].average()'.format(i+1))

            exec('evoked_TriSnumSfsSis{} = epoch[num9].average()'.format(i+1))
            exec('evoked_TriSnumSfsLis{} = epoch[num10].average()'.format(i+1))
            exec('evoked_TriSnumLfsSis{} = epoch[num11].average()'.format(i+1))
            exec('evoked_TriSnumLfsLis{} = epoch[num12].average()'.format(i+1))
            exec('evoked_TriLnumSfsSis{} = epoch[num13].average()'.format(i+1))
            exec('evoked_TriLnumSfsLis{} = epoch[num14].average()'.format(i+1))
            exec('evoked_TriLnumLfsSis{} = epoch[num15].average()'.format(i+1))
            exec('evoked_TriLnumLfsLis{} = epoch[num16].average()'.format(i+1))

            fileNum = fileNum + 1
            i = i + 1

    evokedAll_CirSnumSfsSis = []
    evokedAll_CirSnumSfsLis = []
    evokedAll_CirSnumLfsSis = []
    evokedAll_CirSnumLfsLis = []
    evokedAll_CirLnumSfsSis = []
    evokedAll_CirLnumSfsLis = []
    evokedAll_CirLnumLfsSis = []
    evokedAll_CirLnumLfsLis = []

    evokedAll_TriSnumSfsSis = []
    evokedAll_TriSnumSfsLis = []
    evokedAll_TriSnumLfsSis = []
    evokedAll_TriSnumLfsLis = []
    evokedAll_TriLnumSfsSis = []
    evokedAll_TriLnumSfsLis = []
    evokedAll_TriLnumLfsSis = []
    evokedAll_TriLnumLfsLis = []

    for i in range(fileNum):# range(fileNum): #range(fileNum)[0,1,2,3,6,7,8,9,10,11,12,13,14]
        exec('evokedAll_CirSnumSfsSis.append(evoked_CirSnumSfsSis{})'.format(i + 1))
        exec('evokedAll_CirSnumSfsLis.append(evoked_CirSnumSfsLis{})'.format(i + 1))
        exec('evokedAll_CirSnumLfsSis.append(evoked_CirSnumLfsSis{})'.format(i + 1))
        exec('evokedAll_CirSnumLfsLis.append(evoked_CirSnumLfsLis{})'.format(i + 1))
        exec('evokedAll_CirLnumSfsSis.append(evoked_CirLnumSfsSis{})'.format(i + 1))
        exec('evokedAll_CirLnumSfsLis.append(evoked_CirLnumSfsLis{})'.format(i + 1))
        exec('evokedAll_CirLnumLfsSis.append(evoked_CirLnumLfsSis{})'.format(i + 1))
        exec('evokedAll_CirLnumLfsLis.append(evoked_CirLnumLfsLis{})'.format(i + 1))

        exec('evokedAll_TriSnumSfsSis.append(evoked_TriSnumSfsSis{})'.format(i + 1))
        exec('evokedAll_TriSnumSfsLis.append(evoked_TriSnumSfsLis{})'.format(i + 1))
        exec('evokedAll_TriSnumLfsSis.append(evoked_TriSnumLfsSis{})'.format(i + 1))
        exec('evokedAll_TriSnumLfsLis.append(evoked_TriSnumLfsLis{})'.format(i + 1))
        exec('evokedAll_TriLnumSfsSis.append(evoked_TriLnumSfsSis{})'.format(i + 1))
        exec('evokedAll_TriLnumSfsLis.append(evoked_TriLnumSfsLis{})'.format(i + 1))
        exec('evokedAll_TriLnumLfsSis.append(evoked_TriLnumLfsSis{})'.format(i + 1))
        exec('evokedAll_TriLnumLfsLis.append(evoked_TriLnumLfsLis{})'.format(i + 1))

    # combine fileNum (number) evoked epochs
    evokedAvg_CirSnumSfsSis = mne.combine_evoked(evokedAll_CirSnumSfsSis, weights='nave')
    evokedAvg_CirSnumSfsLis = mne.combine_evoked(evokedAll_CirSnumSfsLis, weights='nave')
    evokedAvg_CirSnumLfsSis = mne.combine_evoked(evokedAll_CirSnumLfsSis, weights='nave')
    evokedAvg_CirSnumLfsLis = mne.combine_evoked(evokedAll_CirSnumLfsLis, weights='nave')
    evokedAvg_CirLnumSfsSis = mne.combine_evoked(evokedAll_CirLnumSfsSis, weights='nave')
    evokedAvg_CirLnumSfsLis = mne.combine_evoked(evokedAll_CirLnumSfsLis, weights='nave')
    evokedAvg_CirLnumLfsSis = mne.combine_evoked(evokedAll_CirLnumLfsSis, weights='nave')
    evokedAvg_CirLnumLfsLis = mne.combine_evoked(evokedAll_CirLnumLfsLis, weights='nave')

    evokedAvg_TriSnumSfsSis = mne.combine_evoked(evokedAll_TriSnumSfsSis, weights='nave')
    evokedAvg_TriSnumSfsLis = mne.combine_evoked(evokedAll_TriSnumSfsLis, weights='nave')
    evokedAvg_TriSnumLfsSis = mne.combine_evoked(evokedAll_TriSnumLfsSis, weights='nave')
    evokedAvg_TriSnumLfsLis = mne.combine_evoked(evokedAll_TriSnumLfsLis, weights='nave')
    evokedAvg_TriLnumSfsSis = mne.combine_evoked(evokedAll_TriLnumSfsSis, weights='nave')
    evokedAvg_TriLnumSfsLis = mne.combine_evoked(evokedAll_TriLnumSfsLis, weights='nave')
    evokedAvg_TriLnumLfsSis = mne.combine_evoked(evokedAll_TriLnumLfsSis, weights='nave')
    evokedAvg_TriLnumLfsLis = mne.combine_evoked(evokedAll_TriLnumLfsLis, weights='nave')


    # Save evoked data    
    evokednumS = mne.combine_evoked(evokedAll_CirSnumSfsSis+evokedAll_CirSnumSfsLis+evokedAll_CirSnumLfsSis+evokedAll_CirSnumLfsLis
                                   + evokedAll_TriSnumSfsSis + evokedAll_TriSnumSfsLis + evokedAll_TriSnumLfsSis + evokedAll_TriSnumLfsLis, weights='nave')
    evokednumS.save(pj(savePath, 'Smallnum' + '_ave.fif'))
    evokednumL = mne.combine_evoked(evokedAll_CirLnumSfsSis+evokedAll_CirLnumSfsLis+evokedAll_CirLnumLfsSis+evokedAll_CirLnumLfsLis
                                   + evokedAll_TriLnumSfsSis + evokedAll_TriLnumSfsLis + evokedAll_TriLnumLfsSis + evokedAll_TriLnumLfsLis, weights='nave')
    evokednumL.save(pj(savePath, 'Largenum' + '_ave.fif'))
    
    
    
    
    
    
                                    