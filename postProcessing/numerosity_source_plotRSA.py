import os.path as op
from matplotlib import pyplot as plt
import mne
import mne_rsa
from os.path import join as pj

mne.set_log_level(False)  # Be less verbose
mne.viz.set_3d_backend('mayavi')
import joblib
rootDir = r'E:\temp'
subjid = 'subj004'
RDMname = ['FieldArea','Numerosity','ItemArea','Shape']
subjects_dir = pj(rootDir,'surf')


for i in [0]: #range(len(RDMname))
    rsaName = pj(rootDir,subjid,'rsa_run1'+RDMname[i]+'.stc')
    rsa_vals = joblib.load(rsaName)

    # Find the searchlight patch with highest RSA score
    peak_vertex, peak_time = rsa_vals.get_peak(vert_as_index=True)

    '''
    surfer_kwargs = dict(
    hemi='rh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[0, 0.01, 0.05]), views='lateral',
    initial_time=peak_time, time_unit='s', size=(800, 800), smoothing_steps=10)
    brain = rsa_vals.plot(**surfer_kwargs)
    '''

    # Plot the result at the timepoint where the maximum RSA value occurs.
    brain = rsa_vals.plot(subjid, subjects_dir=subjects_dir, hemi='both', initial_time=peak_time,time_viewer=True) #initial_time=peak_time
    #rsa_vals.add_annotation('HCPMMP1_combined', borders=2, subjects_dir=subjects_dir)

'''
# brain = rsa_vals.plot(subjects_dir=subjects_dir, hemi='both', initial_time=0, clim=dict(kind='value', lims=[3, 6, 9]),time_viewer=True)
movieFname = pj(rootDir,'my.mp4')
brain.save_movie(filename=movieFname,time_dilation=20, tmin=0, tmax=0.6,framerate=24, interpolation='linear',time_viewer=True)
print('All Done')
'''
# plot the max vertex
plt.figure()
plt.plot(rsa_vals.times, rsa_vals.data[peak_vertex])
plt.xlabel('Time (s)')
plt.ylabel('Kendall-Tau (alpha)')
plt.title(f'RSA values at vert {peak_vertex}')
plt.show()

