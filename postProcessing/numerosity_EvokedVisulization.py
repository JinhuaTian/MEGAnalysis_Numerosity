# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 21:16:07 2021

@author: tclem
"""

import mne
import numpy as np
import os
from os.path import join as pj
import matplotlib
# matplotlib.use('TkAgg')
import joblib
from mne.preprocessing import annotate_movement, compute_average_dev_head_t

#filepath = '/nfs/s2/userhome/tianjinhua/workingdir/meg/numerosity/gu_wenyu/preprocessed/filterEpochOnly_run8_tsss.fif'     run1_tsss.fif
rootDir = 'E:/temp'
subjName = 'subj004'

Snum_evoked=mne.read_evokeds(pj(rootDir, subjName, 'preprocessed', 'Smallnum' + '_ave.fif'))
Snum_evoked= mne.combine_evoked(Snum_evoked,weights='nave')
Lnum_evoked=mne.read_evokeds(pj(rootDir, subjName, 'preprocessed', 'Largenum' + '_ave.fif'))
Lnum_evoked= mne.combine_evoked(Lnum_evoked,weights='nave')
surfDir =  r'E:\temp\surf'

newSamplingRate = 100

for patient_ID in ["subj004"]: #, "subj006"
    subjPath = pj(rootDir, patient_ID)
    surfPath = pj(surfDir,patient_ID)
    fname_trans = pj(surfPath, patient_ID+'-trans.fif')

    surf_src = pj(surfPath, 'bem', patient_ID+'_watershed_icoNone_bem_pial_surf-src.fif')

    bem_sol = pj(surfPath, 'bem', patient_ID+'_watershed_icoNone_bem-sol.fif')

    #read and max filter the raw file:
    for file in os.listdir(subjPath):
        if 'Noise-default_Noise' in file:
            fname_empty_room = pj(subjPath, file)
            raw_empty_room = mne.io.read_raw_ctf(fname_empty_room, preload=True)
    mf_kwargs = dict(origin=(0., 0., 0.), st_duration=10.)
    raw_empty_room = mne.preprocessing.maxwell_filter(raw_empty_room, **mf_kwargs, coord_frame="meg")

    fpath_sourced = pj(subjPath, 'sourced')
    if not os.path.exists(fpath_sourced):
        os.makedirs(fpath_sourced)
    
    epochNum=1
    for prepEpoch in [Snum_evoked,Lnum_evoked]:



        fwd = mne.make_forward_solution(prepEpoch.info, trans=fname_trans, src=surf_src, bem=bem_sol,
                                        meg=True, eeg=False, mindist=5.0, n_jobs=4) # elkta use meg = True

        fname_fwd = pj(subjPath, 'Evoked' + str(epochNum)+'_fwd.fif')

        mne.write_forward_solution(fname_fwd, fwd, overwrite=True) # save forward solution for each run

        noise_cov = mne.compute_raw_covariance(raw_empty_room, method=['empirical', 'shrunk'])

        # test1 = mne.read_trans(fname_trans)

        # test2 = mne.read_bem_surfaces(bem_sol, patch_stats=False, s_id=None, on_defects='raise')

        #fname_meg_raw = os.path.join(dataPath, patient_ID, 'preprocessed', 'filterEpochICAMR_{}.ds'.format(runs+1))
        #evoked = mne.read_epochs(fname_meg_raw)

        inv_operator = mne.minimum_norm.make_inverse_operator(
            prepEpoch.info, fwd, noise_cov, loose=0.2, depth=0.8)

        epochs_stc = mne.minimum_norm.apply_inverse_epochs(prepEpoch, inv_operator, lambda2=0.1111,method='dSPM')

        fname_epochs_stc = pj(fpath_sourced, 'Evoked' + str(epochNum) +'.stc')  # or save as stc file ?
        
        with open(fname_epochs_stc, 'wb') as f:
            joblib.dump(epochs_stc, f)
        f.close()

        del epochs_stc, prepEpoch
        print(patient_ID, 'is finished')
        epochNum = epochNum+1
    del noise_cov, fwd

