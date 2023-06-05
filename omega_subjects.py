# %% Imports
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import h5io
import numpy as np
import mne

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
plt.savefig('repartition_subjects_group.png')

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

epochs = mne.read_epochs('/storage/store3/work/kachardo/derivatives/omega/sub-0221/ses-01/meg/sub-0221_ses-01_task-rest_proc-clean_epo.fif')
print(epochs.info['ch_names'])
print(len(epochs.info['ch_names']))

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

features_log = pd.concat([features_ses01[features_ses01['ok']=='OK'], features_ses02[features_ses02['ok'] == 'OK']], ignore_index=True)
features_log = features_log.sort_values(by = 'subject')
# %%

features_log.to_csv(os.path.join(deriv_root,'feature_fb_covs_rest-log.csv'))

#%% Running the benchmark only for healthy subjects
# Concat the files of the 2 sessions in one

features_ses01 = h5io.read_hdf5(
            deriv_root / 'features_fb_covs_rest_ses-01.h5')

features_ses02 = h5io.read_hdf5(
            deriv_root / 'features_fb_covs_rest_ses-02.h5')

features = {**features_ses01, **features_ses02}
features = dict(sorted(features.items()))

out_fname = deriv_root / 'features_fb_covs_rest.h5'

# %%

h5io.write_hdf5(
            out_fname,
            features,
            overwrite=True
        )


# python compute_benchmark_age_prediction.py --n_jobs 1 -d omega -b filterbank-riemann
# -> ./results/benchmark-filterbank-riemann_dataset-omega_ys.csv et ./results/benchmark-filterbank-riemann_dataset-omega.csv

# %% Age distribution

A = all_subjects.loc[all_subjects['group'] == 'Control', 'age']
B = all_subjects.loc[all_subjects['group'] == 'Parkinson', 'age']
C = all_subjects.loc[all_subjects['group'] == 'Chronic Pain', 'age']

plt.hist(A, alpha=0.5, label='Control')
plt.hist(B, alpha=0.5, label='Parkinson')
plt.hist(C, alpha=0.5, label='Chronic Pain')

plt.title('Age Distribution by Group')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Group')

plt.savefig('repartition_subjects_age.png')


# %% Visualisation of true age vs predicted age

results = pd.read_csv("/storage/store3/work/kachardo/brain-age-benchmark-paper/results/benchmark-filterbank-riemann_dataset-omega_ys.csv", index_col=0)
plt.scatter(range(len(list(results['y_true']))), list(results['y_true']), label = 'True ages', )
plt.scatter( range(len(list(results['y_pred']))), list(results['y_pred']), label = 'Predicted ages')

plt.title('True age vs Predicted age for each participant')
plt.xlabel('Subject number')
plt.ylabel('Age')
plt.legend(title='Group')

plt.savefig('repartition_subjects_true_pred.png')

plt.show()

A = results['y_true']
B = results['y_pred']

bins = range(20,90,1)

plt.hist(A, bins, alpha=0.5, label='True ages')
plt.hist(B, bins, alpha=0.5, label='Predicted ages')

plt.title('Distributions of true ages and predicted ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Group')

plt.savefig('repartition_subjects_true_pred2.png')

# %% Number of subjects with age <= 50 and > 50

counts_age_50 = np.unique(np.where(all_subjects['age'] <= 50, 0, 1), return_counts = True)

plt.bar(counts_age_50[0],counts_age_50[1], label = ['age <= 50', 'age > 50'], color = ['green', 'red'])
plt.legend()
plt.xticks([], [])
plt.title('Number of subjects with age <= 50 and > 50')

plt.savefig('repartition_subjects_age_50.png')

percentage_sup_50 = counts_age_50[1][1] / counts_age_50[1].sum() * 100
print('~'+str(np.floor(percentage_sup_50)) + '% of the subjects have age > 50')

# %% Number of control subjects with age <= 50 and > 50

counts_age_50_control = np.unique(np.where(all_subjects[all_subjects['group'] == 'Control']['age'] <= 50, 0, 1), return_counts = True )

plt.bar(counts_age_50_control[0],counts_age_50_control[1], label = ['age <= 50', 'age > 50'], color = ['green', 'red'])
plt.legend()
plt.xticks([], [])
plt.title('Number of control subjects with age <= 50 and > 50')

plt.savefig('repartition_control_subjects_age_50.png')

percentage_sup_50_control = counts_age_50_control[1][1] / counts_age_50_control[1].sum() * 100
print('~'+str(np.floor(percentage_sup_50_control)) + '% of the control subjects have age > 50')

# %% Scatterplot (predictions for all subjects)

results = pd.read_csv("/storage/store3/work/kachardo/brain-age-benchmark-paper/results/benchmark-filterbank-riemann_dataset-omega_ys.csv", index_col=0)
plt.scatter(range(len(list(results['y_true']))), list(results['y_true']), label = 'True ages', )
plt.scatter( range(len(list(results['y_pred']))), list(results['y_pred']), label = 'Predicted ages')

plt.title('True age vs Predicted age for each participant')
plt.xlabel('Subject number')
plt.ylabel('Age')
plt.legend(title='Group')

plt.savefig('repartition_subjects_true_pred.png')

plt.show()

# %% Scatterplot 2
results = pd.read_csv("/storage/store3/work/kachardo/brain-age-benchmark-paper/results/benchmark-filterbank-riemann_dataset-omega_ys.csv", index_col=0)
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

fig, ax = plt.subplots()

ax.set_xlim(20, 90)
ax.set_ylim(20,90)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(10))

ax.xaxis.set_minor_locator(AutoMinorLocator(10))
ax.yaxis.set_minor_locator(AutoMinorLocator(10))

ax.grid(which='major', alpha = 0.2)
ax.grid(which='minor', alpha = 0.2)

plt.scatter(list(results['y_true']), list(results['y_pred']), color = 'orange', label = "subject", marker = 'x', alpha = 0.5)
plt.plot(range(22,85), range(22,85), color = 'red', label = "x=y")
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title('True age vs Predicted age')
plt.legend()

plt.savefig('scatter_predictions.png')

plt.show()

# %% Find subject with bad predictions' reports
results = results.reset_index()
# %%
diff = np.abs(results['y_true'] - results['y_pred'])
diff_idx = diff.sort_values(ascending = False).index
all_subjects[all_subjects['group'] == 'Control'].iloc[diff_idx[:10],:]

# %% Find subject's reports with good predictions

diff_idx = diff.sort_values(ascending = True).index
all_subjects[all_subjects['group'] == 'Control'].iloc[diff_idx[:15],:]

# %% Compare nb of clean epochs after and before autoreject

bad_sub_dir = os.path.join(deriv_root,"sub-0258/ses-02/meg")
bad2_sub_dir = os.path.join(deriv_root,"sub-0312/ses-02/meg")
good_sub_dir = os.path.join(deriv_root,"sub-0470/ses-02/meg")
good2_sub_dir = os.path.join(deriv_root,"sub-CONP0106/ses-02/meg")

bad_before = os.path.join(bad_sub_dir,"sub-0258_ses-02_task-rest_proc-clean_epo.fif")
bad_after = os.path.join(bad_sub_dir,"sub-0258_ses-02_task-rest_proc-autoreject_epo.fif")

bad2_before = os.path.join(bad2_sub_dir,"sub-0312_ses-02_task-rest_proc-clean_epo.fif")
bad2_after = os.path.join(bad2_sub_dir,"sub-0312_ses-02_task-rest_proc-autoreject_epo.fif")

good_before = os.path.join(good_sub_dir,"sub-0470_ses-02_task-rest_proc-clean_epo.fif")
good_after = os.path.join(good_sub_dir,"sub-0470_ses-02_task-rest_proc-autoreject_epo.fif")

good2_before = os.path.join(good2_sub_dir,"sub-CONP0106_ses-02_task-rest_proc-clean_epo.fif")
good2_after = os.path.join(good2_sub_dir,"sub-CONP0106_ses-02_task-rest_proc-autoreject_epo.fif")

# %%

epochs_bad_before = mne.read_epochs(bad_before)
epochs_bad_after = mne.read_epochs(bad_after)

epochs_bad2_before = mne.read_epochs(bad2_before)
epochs_bad2_after = mne.read_epochs(bad2_after)

epochs_good_before = mne.read_epochs(good_before)
epochs_good_after = mne.read_epochs(good_after)

epochs_good2_before = mne.read_epochs(good2_before)
epochs_good2_after = mne.read_epochs(good2_after)

# %% 

print("bad : sub-0258")
print(epochs_bad_before)
print(epochs_bad_after)

print("bad : sub-0312")
print(epochs_bad2_before)
print(epochs_bad2_after)

print("good : sub-0470")
print(epochs_good_before)
print(epochs_good_after)

print("good : sub-CONP0106")
print(epochs_good2_before)
print(epochs_good2_after)

# %%

print(epochs_bad_before.get_channel_types())
print(len(epochs_bad_before.get_channel_types()))
print(epochs_bad_after.get_channel_types())
print(len(epochs_bad_after.get_channel_types()))

# %% Print PSD of all the subjects (after autoreject)

fig, axs = plt.subplots(nrows=82, ncols=2, layout='constrained', figsize=(20,200))
i = 0
for subject in os.listdir(deriv_root):
    if subject.startswith('sub'):
        id = subject[4:]
        if subjects_data[subjects_data['subject_id']==id]['group'].iloc[0] == 'Control':
            session = subjects_data[subjects_data['subject_id']==id]['session'].iloc[0]
            epoch_file = os.path.join(deriv_root,subject, "ses-0"+str(session), "meg","sub-"+str(id)+"_ses-0"+str(session)+"_task-rest_proc-autoreject_epo.fif")
            epoch = mne.read_epochs(epoch_file)
            axs.flat[i].set_ylim([-10,120])
            axs.flat[i].set_xlabel(subject)
            epoch.compute_psd().plot(axes = axs.flat[i])
            i +=1

plt.show()
fig.savefig('omega_subjects_control_psd.png')