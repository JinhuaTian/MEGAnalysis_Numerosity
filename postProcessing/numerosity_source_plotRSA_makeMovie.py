import os
import os.path as op
from matplotlib import pyplot as plt
import mne
import mne_rsa
from os.path import join as pj
import numpy as np
mne.set_log_level(False)  # Be less verbose
mne.viz.set_3d_backend('mayavi')
import joblib
imgdir = 'thepic'
rootDir = r'E:\temp\ssRSA\subj006_100_bf'
method = 'bf'
subjid = 'subj006'
RDMname = ['FieldArea','Numerosity','ItemArea','Shape']
subjects_dir = pj('E:/temp','heads')
for i in [0,1,2]: #range(4): #range(len(RDMname)):
    rsaName = pj(rootDir,'rsa_'+method+'100'+RDMname[i]+'.stc')
    rsa_vals = joblib.load(rsaName)
    peak_vertex, peak_time = rsa_vals.get_peak(vert_as_index=True)
    max = np.max(rsa_vals.data)
    for time in range(50):
        # Find the searchlight patch with highest RSA score
        plotime = time/100+0.01
        brain = rsa_vals.plot(views='lat', hemi='split', size=(800, 400), subject=subjid,
            subjects_dir=subjects_dir, initial_time=plotime,title=RDMname[i])  #lims=[max*(-1),0,max] clim=dict(kind='value', lims=[-0.002,0,0.002])
        #brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
        #        font_size=14)
        imgDir = pj(rootDir,method+RDMname[i])
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
        imgPath = pj(imgDir,str(time)+'.png')
        brain.save_image(imgPath)
        brain.close()

print('All Done')