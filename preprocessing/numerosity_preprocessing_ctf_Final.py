'''
Script for MEG data preprocessing
# preprocessing procedure:
1.load data
2.run SSP
3.filter:notch, high pass, low pass
4.run ICA  # before epoch, 1 Hz highpass suggested
5.epoch data (auto rejection, baseline correction)

# file2:
6.Annotate and delete bad epochs manually

# post prep
7.average epochs
8.concatenate data at subject level
9.source reconstruction
10.decoding
'''
import numpy as np
import os, sys
from os.path import join as pj
import time
from mne.io import read_raw_ctf

import matplotlib

matplotlib.use('TkAgg')  # Qt5Agg #'TkAgg'
'''
oldMne = '/usr/local/neurosoft/anaconda3/lib/python3.8/site-packages/mne'
sys.path.remove(oldMne)
currMne = '/nfs/s2/userhome/tianjinhua/workingdir/code'
sys.path.append(currMne)
'''
import mne

rootDir = '/nfs/s2/userhome/tianjinhua/workingdir/meg/numerosity2'
subjName = ['subj016']
# 'subj002','subj003','subj004','subj005','subj006','subj007','subj008','subj009','subj010'
taskName = 'raw'

# filter parameters
freqs = np.arange(50, 200, 50)
highcutoff = 0.1  # ICA recommandation
lowcutoff = 40
newSamplingRate = 500
reject = dict(mag=4e-12)  # eog=250e-6, there is no EOG!

from mne.preprocessing import (create_eog_epochs, create_ecg_epochs, compute_proj_ecg, compute_proj_eog)

for subj in subjName:
    rawDir = pj(rootDir, subj, taskName)
    # name and makedir the save path
    savePath = pj(rootDir, subj, 'preprocessed')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    # walk through subj path, filter, epoch and save epochd ata
    for file in os.listdir(rawDir):
        if 'G15BNU' in file:
            fileName = 'filterEpochICA_' + file
            savePath2 = pj(savePath, fileName)
            if not os.path.exists(savePath2):
                # rawName = raw + str(fileNum)
                filepath = pj(rawDir, file)
                raw = read_raw_ctf(filepath, preload=True)  # allow_maxshield=True,

                # --------------------------------------
                # 1.1 load data, run SSP, filter
                # --------------------------------------
                raw.apply_gradient_compensation(0)  # must un-do software compensation first
                mf_kwargs = dict(origin=(0., 0., 0.), st_duration=10.)
                raw = mne.preprocessing.maxwell_filter(raw, **mf_kwargs)
                '''
                # 1. To do a dipole fit, let¡¯s use the covariance provided by the empty room recording.
                raw_erm = read_raw_ctf(erm_path).apply_gradient_compensation(0)
                mf_kwargs = dict(origin=(0., 0., 0.), st_duration=10.)
                raw_erm = mne.preprocessing.maxwell_filter(raw_erm, coord_frame='meg',**mf_kwargs)
                '''

                # filter the data: notch, high pass, low pass
                # meg_picks = mne.pick_types(raw.info, mag=True, eeg=False, eog=False)
                raw = raw.notch_filter(freqs=freqs, picks='mag', method='spectrum_fit',
                                       filter_length="auto", phase="zero", verbose=True)  # n_jobs=4,
                raw = raw.filter(l_freq=highcutoff, h_freq=None)
                raw = raw.filter(l_freq=None, h_freq=lowcutoff)
                # raw.plot(block = True)

                # --------------------------------------
                # 1.2 run ICA and reject artifact components
                # just remove eye blink, horizontal eye movement and muscle
                # --------------------------------------
                from mne.preprocessing import ICA  # , create_eog_epochs, create_ecg_epochs,corrmap

                ica = ICA()  # n_components=90,
                ica.fit(raw)
                ica.plot_sources(raw, show_scrollbars=True)  # ,block=True
                ica.plot_components()  # 0 14 17 21 32; 0 1 6; 2 10 32

                # reject ica components from input
                ica_rej = input()
                bad_comps = ica_rej.split(" ")
                bad_comps = [int(bad_comps[i]) for i in range(len(bad_comps))]  # transform str to number

                # ica.exclude = list(ica_rej)
                ica_rej = ica.apply(raw, exclude=bad_comps)
                # plot psd
                ica_rej.plot_psd(fmax=100)

                # --------------------------------------
                # 1.3 Epoch and reject bad epochs
                # --------------------------------------
                # peak-to-peak amplitude rejection parameters
                # select events and epoch data
                events = mne.find_events(ica_rej, stim_channel='UPPT001', shortest_event=2, min_duration=0.005)

                # modify events, set the events prior to key press to 99
                for i in range(events.shape[0]):
                    if events[i, 2] == 101:
                        events[i - 1, 2] = 99

                events_no_1back = mne.pick_events(events, exclude=[99, 101])

                # epoch data: select events, remove the first epoch.
                ica_rej = mne.Epochs(ica_rej, events_no_1back, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), reject=reject,
                                     preload=True, detrend=1, verbose=True)  # events_no_1back[1:, :]
                ica_rej.apply_baseline((-0.2, 0))
                # save ICA rejected and epoched files
                ica_rej.save(savePath2, overwrite=True)

                # --------------------------------------
                # 1.4 Reject epochs manually
                # --------------------------------------
                # select and annotate bad epoch
                fig = ica_rej.plot(picks='mag', block=True)  # stop until plot closed, could run in ipython #fig.canvas.key_press_event('a')
                # apply bad epoch
                ica_rej.drop_bad()
                # save the manual rejected file
                fileName = 'filterEpochICAMR_' + file
                tempSavename = pj(savePath, fileName)
                # save manually inspected data
                ica_rej.save(tempSavename, overwrite=True)

                del raw, ica_rej

print('All Done')