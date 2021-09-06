# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:27:23 2021

@author: tclem

Channel-wise searchlight
"""
import mne
import mne_rsa
import os
from os.path import join as pj
from mne.transforms import Transform
import numpy as np
# basic info
rootDir = r'E:\temp'
epochs_list = []
subjid = 'subj004'
savePath = pj(rootDir,subjid,'preprocessed')
newSamplingRate = 100

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

eventsFs = epochs_all.events

# make stimulus RDM
eventMatrix =  np.loadtxt('C:/Users/tclem/Documents/GitHub/MEGAnalysis_Numerosity/postProcessing/STI.txt')

eventNum = 80
for i in range (eventNum):
    eventsFs[eventsFs[:,2]==(i+1),2] = eventMatrix[eventMatrix[:,4]==(i+1),0]

dsmFs =  mne_rsa.compute_dsm(eventsFs[:,2],metric='sqeuclidean')
# Plot the DSM
fig = mne_rsa.plot_dsms(dsmFs, title='Field size DSM')

rsa_result = mne_rsa.rsa_epochs(
    epochs_all,                           # The EEG data
    eventsFs,                          # The model DSM
    epochs_dsm_metric='sqeuclidean',  # Metric to compute the EEG DSMs
    rsa_metric='kendall-tau-a',       # Metric to compare model and EEG DSMs
    spatial_radius=45,                # Spatial radius of the searchlight patch
    temporal_radius=0.05,             # Temporal radius of the searchlight path
    tmin=0, tmax=0.6,             # To save time, only analyze this time interval
    n_jobs=4,                         # Only use one CPU core. Increase this for more speed.
    n_folds=None,
    verbose=False)                    # Set to True to display a progress bar

labeNum = 2189
count=0
for x in range(labeNum):
    for y in range(x+1,labeNum):
        #if x != y: #x + y < 80:
        count = count+1

