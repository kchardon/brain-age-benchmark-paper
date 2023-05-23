# %% Imports
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5io
import numpy as np

bids_root = pathlib.Path('/storage/store2/data/Omega')
deriv_root = pathlib.Path('/storage/store3/work/kachardo/derivatives/omega')

# %% Participants data
participants_file = os.path.join(bids_root, "participants.tsv")
all_subjects = pd.read_csv(participants_file, sep = '\t')
subjects_id = all_subjects['participant_id'].str.extract(r'sub-([\d\w]+)')

# %% Get the session and run to use for each subject
subjects_data = pd.DataFrame(columns = ['subject_id', 'group', 'session', 'run'])

for subject in os.listdir(bids_root):
    if subject.startswith('sub'):
        find = False
        for ses in range(1,5):
            if find :
                break
            for run in range(1,5):
                if find :
                    break
                dir_path = pathlib.Path(os.path.join(
                bids_root,
                subject,
                'ses-0'+str(ses)+'/meg',
                subject + '_ses-0'+str(ses)+'_task-rest_run-0'+str(run)+'_meg.ds'))
                if dir_path.exists():
                    new_row = {'subject_id' : subject[4:], 'group' : all_subjects[subjects_id[0] == subject[4:]]['group'][len(subjects_data)], 'session' : ses, 'run' : run}
                    subjects_data.loc[len(subjects_data)] = new_row
                    find = True

# %% Save the group, session and run for each subject as CSV
subjects_data.to_csv('omega_subjects.csv', index=False)

#%% Number of subjects per session and run
subjects_data.groupby(['session', 'run']).count()

#%% Get a list of subject for each (session, run)

list11 = list(subjects_data[(subjects_data['session'] == 1) & (subjects_data['run'] == 1)]['subject_id'])
list12 = list(subjects_data[(subjects_data['session'] == 1) & (subjects_data['run'] == 2)]['subject_id'])
list13 = list(subjects_data[(subjects_data['session'] == 1) & (subjects_data['run'] == 3)]['subject_id'])

list21 = list(subjects_data[(subjects_data['session'] == 2) & (subjects_data['run'] == 1)]['subject_id'])
list22 = list(subjects_data[(subjects_data['session'] == 2) & (subjects_data['run'] == 2)]['subject_id'])
list23 = list(subjects_data[(subjects_data['session'] == 2) & (subjects_data['run'] == 3)]['subject_id'])
# %% Preprocessing on each list of subjects with mne_bids_pipeline and config_omega_meg.py
# 11, 21, 12, 13, 22, 23
# -> task-rest_proc-clean_epo.fif (global file deleted)


# Plot distribution of groups

plt.bar([0,1,2], list(subjects_data.groupby(['group']).count().iloc[:,0]), color = ['red','blue','yellow'], label = list(subjects_data.groupby(['group']).count().index))
plt.title("Repartition of subjects by groups")
plt.legend()
#plt.show()
plt.savefig('repartition_subjects.png')

# %% Verify is all the subjects have been preprocessed

sub_preprocess = []

for subject in os.listdir(deriv_root):
    if subject.startswith('sub'):
        session = int(subjects_data[subjects_data['subject_id']==subject[4:]]['session'])
        count = 0
        for file in os.listdir(pathlib.Path(os.path.join(deriv_root, subject, 'ses-0'+str(session), 'meg'))):
            count += 1
        if count >= 6:
            sub_preprocess.append(subject[4:])

print(set(subjects_data['subject_id']) - set(sub_preprocess))
print(set(sub_preprocess) - set(subjects_data['subject_id']))
        

# %% Find the good channel names

import mne

epochs = mne.read_epochs('/storage/store3/work/kachardo/derivatives/omega/sub-0221/ses-01/meg/sub-0221_ses-01_task-rest_proc-clean_epo.fif')
epochs.info['ch_names']

# %% Preprocessing with brain-age-benchmark

# Autoreject for each session
# python compute_autoreject.py --n_jobs 40 -d omega
# 2, 1
# -> task-rest_proc-autoreject_epo.fif and a global file autoreject_log.csv (with _ses-01 or _ses-02)



# Verify if all the subjects have been preprocessed with autoreject

autoreject_ses01 = pd.read_csv(os.path.join(deriv_root, 'autoreject_log_ses-01.csv'), index_col=0)
autoreject_ses02 = pd.read_csv(os.path.join(deriv_root, 'autoreject_log_ses-02.csv'), index_col=0)

set01 = set(autoreject_ses01[autoreject_ses01['ok'] == 'OK']['subject'])
set01.update(set(autoreject_ses02[autoreject_ses02['ok'] == 'OK']['subject']))

sub_autoreject = set()

for subject in set01:
    sub_autoreject.add(subject[4:])

print(set(subjects_data['subject_id']) - sub_autoreject)
print(sub_autoreject - set(subjects_data['subject_id']))

# %% Computing the covariances (filterbank-riemann) for each session
# python compute_features.py --n_jobs 40 -d omega -t fb_covs
# 1, 2
# -> feature_fb_covs_rest-log.csv and features_fb_covs_rest.h5(both with _ses-01 or _ses-02))



# See if all the subjects have their covariance 

features_ses01 = pd.read_csv(os.path.join(deriv_root, 'feature_fb_covs_rest-log_ses-01.csv'), index_col=0)
features_ses02 = pd.read_csv(os.path.join(deriv_root, 'feature_fb_covs_rest-log_ses-02.csv'), index_col=0)

set01 = set(features_ses01[features_ses01['ok'] == 'OK']['subject'])
set01.update(set(features_ses02[features_ses02['ok'] == 'OK']['subject']))

sub_features = set()

for subject in set01:
    sub_features.add(subject[4:])

print(set(subjects_data['subject_id']) - sub_features)
print(sub_features - set(subjects_data['subject_id']))



#%% Save log files of twe two sessions in one
features_log = pd.concat([features_ses01[features_ses01['ok']=='OK'], features_ses02[features_ses02['ok'] == 'OK']])
features_log.to_csv(os.path.join(deriv_root,'feature_fb_covs_rest-log.csv'))



#%% Running the benchmark only for healthy subjects
# Concat the files of the 2 sessions in one

features_ses01 = h5io.read_hdf5(
            deriv_root / 'features_fb_covs_rest_ses-01.h5')

features_ses02 = h5io.read_hdf5(
            deriv_root / 'features_fb_covs_rest_ses-02.h5')

features = {**features_ses01, **features_ses02}

out_fname = deriv_root / 'features_fb_covs_rest.h5'

h5io.write_hdf5(
            out_fname,
            features,
            overwrite=True
        )


# python compute_benchmark_age_prediction.py --n_jobs 10 -d omega -b filterbank-riemann
# -> 

