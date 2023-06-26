# %% Imports
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import json
import h5io
import numpy as np
import mne

bids_root = pathlib.Path('/storage/store2/data/Omega')
deriv_root = pathlib.Path('/storage/store3/work/kachardo/derivatives/omega')
deriv_root_without_ssp = pathlib.Path('/storage/store3/work/kachardo/derivatives/omega/without_ssp')

# %% Participants data
participants_file = os.path.join(bids_root, "participants.tsv")
all_subjects = pd.read_csv(participants_file, sep = '\t')
subjects_id = all_subjects['participant_id'].str.extract(r'sub-([\d\w]+)')

# %% Subjects data
subjects_data = pd.read_csv('omega_subjects.csv')


# %% Get the session and run to use for each subject
subjects_data = pd.DataFrame(columns = ['subject_id', 'group', 'session', 'run'])
count_not_found = 0

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

                    info_path = pathlib.Path(os.path.join(
                    bids_root,
                    subject,
                    'ses-0'+str(ses)+'/meg',
                    subject + '_ses-0'+str(ses)+'_task-rest_run-0'+str(run)+'_meg.json'))
                    
                    with open(info_path, 'r') as f:
                        data = json.load(f)
                        if data["EOGChannelCount"] == 2 and data["ECGChannelCount"] == 1:
                            new_row = {'subject_id' : subject[4:], 'group' : all_subjects[subjects_id[0] == subject[4:]]['group'].iloc[0], 'session' : ses, 'run' : run}
                            subjects_data.loc[len(subjects_data)] = new_row
                            find = True
                            print('Foud for', subject, "session", ses, "run", run)
        if find == False:
            print("Not found for", subject)
            count_not_found += 1
print("not found for", count_not_found)

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
# mne_bids_pipeline --config config_omega_meg.py --n_jobs 40 --steps=preprocessing
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

# %% Print PSD average of all subjects in one plot
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

fig = plt.figure(figsize = (10,5))
ax = plt.subplot()
fig.add_subplot(ax)
i = 0
colors = list(colors)
color = None

for subject in os.listdir(deriv_root):
    if subject.startswith('sub'):
        id = subject[4:]
        if subjects_data[subjects_data['subject_id']==id]['group'].iloc[0] == 'Control':
            session = subjects_data[subjects_data['subject_id']==id]['session'].iloc[0]
            epoch_file = os.path.join(deriv_root,subject, "ses-0"+str(session), "meg","sub-"+str(id)+"_ses-0"+str(session)+"_task-rest_proc-autoreject_epo.fif")
            epoch = mne.read_epochs(epoch_file)
            if i >= len(colors):
                color = colors[i - len(colors)]
            else:
                color = colors[i]
            epoch.compute_psd().plot(color = color,average = True,axes = ax, ci = None)
            i += 1

ax.set_title('Average PSD for each Control subject')
plt.show()
fig.savefig('omega_subjects_control_psd_average.png')

# %% Print compensation grade

for subject in os.listdir(deriv_root):
    if subject.startswith('sub'):
        id = subject[4:]
        if subjects_data[subjects_data['subject_id']==id]['group'].iloc[0] == 'Control':
            session = subjects_data[subjects_data['subject_id']==id]['session'].iloc[0]
            epoch_file = os.path.join(deriv_root,subject, "ses-0"+str(session), "meg","sub-"+str(id)+"_ses-0"+str(session)+"_task-rest_proc-autoreject_epo.fif")
            epoch = mne.read_epochs(epoch_file, verbose = False)
            if epoch.compensation_grade != 3:
                print(id)
                print('compensation grade = ', epoch.compensation_grade)
            else:
                print(id, "ok")

# %% Raw channels
raw_file = "sub-0221/ses-01/meg/sub-0221_ses-01_task-rest_run-01_meg.ds"
raw_path = os.path.join(bids_root, raw_file)

raw = mne.io.read_raw_ctf(raw_path)
print(raw.info)
raw.set_channel_types({"HEOG": "eog", "VEOG": "eog", "ECG": "ecg"})
raw.info

# %% Processing wits SSP
# 11, 12, 13, 21, 22, 23

# %% Print PSD average of all subjects in one plot

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

fig = plt.figure(figsize = (10,5))
ax = plt.subplot()
fig.add_subplot(ax)
i = 0
colors = list(colors)
color = None

for subject in os.listdir(deriv_root):
    if subject.startswith('sub'):
        id = subject[4:]
        if subjects_data[subjects_data['subject_id']==id]['group'].iloc[0] == 'Control':
            session = subjects_data[subjects_data['subject_id']==id]['session'].iloc[0]
            epoch_file = os.path.join(deriv_root,subject, "ses-0"+str(session), "meg","sub-"+str(id)+"_ses-0"+str(session)+"_task-rest_proc-clean_epo.fif")
            epoch = mne.read_epochs(epoch_file)
            if i >= len(colors):
                color = colors[i - len(colors)]
            else:
                color = colors[i]
            epoch.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax, ci = None)
            i += 1
            
ax.set_title('Average PSD for each Control subject')
plt.show()
# %% Print PSD of all the subjects

fig, axs = plt.subplots(nrows=82, ncols=2, layout='constrained', figsize=(20,200))
i = 0
for subject in os.listdir(deriv_root):
    if subject.startswith('sub'):
        id = subject[4:]
        if subjects_data[subjects_data['subject_id']==id]['group'].iloc[0] == 'Control':
            session = subjects_data[subjects_data['subject_id']==id]['session'].iloc[0]
            epoch_file = os.path.join(deriv_root,subject, "ses-0"+str(session), "meg","sub-"+str(id)+"_ses-0"+str(session)+"_task-rest_proc-clean_epo.fif")
            epoch = mne.read_epochs(epoch_file)
            axs.flat[i].set_ylim([-10,120])
            axs.flat[i].set_xlabel(subject)
            epoch.compute_psd(picks = 'meg').plot(axes = axs.flat[i], picks = 'meg', average = True, ci = None)
            i +=1

plt.show()
fig.savefig('omega_subjects_control_psd_ssp.png')

# %% autoreject after ssp 1, 2
# PSD

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

fig = plt.figure(figsize = (10,5))
ax = plt.subplot()
fig.add_subplot(ax)
i = 0
colors = list(colors)
color = None

for subject in os.listdir(deriv_root):
    if subject.startswith('sub'):
        id = subject[4:]
        if subjects_data[subjects_data['subject_id']==id]['group'].iloc[0] == 'Control':
            session = subjects_data[subjects_data['subject_id']==id]['session'].iloc[0]
            epoch_file = os.path.join(deriv_root,subject, "ses-0"+str(session), "meg","sub-"+str(id)+"_ses-0"+str(session)+"_task-rest_proc-autoreject_epo.fif")
            epoch = mne.read_epochs(epoch_file)
            if i >= len(colors):
                color = colors[i - len(colors)]
            else:
                color = colors[i]
            epoch.compute_psd().plot(color = color,average = True,axes = ax, ci = None)
            i += 1

ax.set_title('Average PSD for each Control subject')
plt.show()
fig.savefig('omega_subjects_control_psd_average_autoreject.png')

# %% Plot PSD with and without SSP
# Before autoreject

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors = list(colors)
color = None

fig = plt.figure(figsize = (10,5))
ax_no_ssp = plt.subplot(2,1,1)
ax_ssp = plt.subplot(2,1,2)
fig.add_subplot(ax_no_ssp)
fig.add_subplot(ax_ssp)

for i, subject in enumerate(list(subjects_data[subjects_data['group'] == 'Control']['subject_id'])):
        session = subjects_data[subjects_data['subject_id']==subject]['session'].iloc[0]
        
        epoch_file_no_ssp = os.path.join(deriv_root_without_ssp,"sub-"+str(subject), "ses-0"+str(session), "meg","sub-"+str(subject)+"_ses-0"+str(session)+"_task-rest_proc-clean_epo.fif")
        epoch_file_ssp = os.path.join(deriv_root,"sub-"+str(subject), "ses-0"+str(session), "meg","sub-"+str(subject)+"_ses-0"+str(session)+"_task-rest_proc-clean_epo.fif")
        
        epoch_no_ssp = mne.read_epochs(epoch_file_no_ssp)
        epoch_ssp = mne.read_epochs(epoch_file_ssp)
        
        if i >= len(colors):
            color = colors[i - len(colors)]
        else:
            color = colors[i]

        epoch_no_ssp.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax_no_ssp, ci = None)
        epoch_ssp.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax_ssp, ci = None)
            
ax_no_ssp.set_title('Average PSD for each Control subject (Without SSP, before autoreject)')
ax_ssp.set_title('Average PSD for each Control subject (With SSP, before autoreject)')
fig

# After autoreject

fig = plt.figure(figsize = (10,5))
ax_no_ssp = plt.subplot(2,1,1)
ax_ssp = plt.subplot(2,1,2)
fig.add_subplot(ax_no_ssp)
fig.add_subplot(ax_ssp)

for i, subject in enumerate(list(subjects_data[subjects_data['group'] == 'Control']['subject_id'])):
        session = subjects_data[subjects_data['subject_id']==subject]['session'].iloc[0]
        
        epoch_file_no_ssp = os.path.join(deriv_root_without_ssp,"sub-"+str(subject), "ses-0"+str(session), "meg","sub-"+str(subject)+"_ses-0"+str(session)+"_task-rest_proc-autoreject_epo.fif")
        epoch_file_ssp = os.path.join(deriv_root,"sub-"+str(subject), "ses-0"+str(session), "meg","sub-"+str(subject)+"_ses-0"+str(session)+"_task-rest_proc-autoreject_epo.fif")
        
        epoch_no_ssp = mne.read_epochs(epoch_file_no_ssp)
        epoch_ssp = mne.read_epochs(epoch_file_ssp)
        
        if i >= len(colors):
            color = colors[i - len(colors)]
        else:
            color = colors[i]

        epoch_no_ssp.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax_no_ssp, ci = None)
        epoch_ssp.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax_ssp, ci = None)
            
ax_no_ssp.set_title('Average PSD for each Control subject (Without SSP, after autoreject)')
ax_ssp.set_title('Average PSD for each Control subject (With SSP, after autoreject)')
fig

# %% Try empty room SSP on data

data_file = os.path.join(deriv_root,"sub-0260/ses-02/meg/","sub-0260_ses-02_task-rest_proc-clean_epo.fif")
raw = mne.read_epochs(data_file)
raw.load_data()

raw_proj = raw.copy()

empty_room_file = os.path.join(deriv_root,"sub-0260/ses-02/meg/","sub-0260_ses-02_task-noise_proc-filt_raw.fif")
empty_room_raw = mne.io.read_raw_fif(empty_room_file)

empty_room_raw.del_proj()

empty_room_projs = mne.compute_proj_raw(empty_room_raw)

raw_proj.add_proj(empty_room_projs)
raw_proj.apply_proj()

fig = plt.figure(figsize = (10,5))
ax = plt.subplot(1,1,1)
fig.add_subplot(ax)

raw.compute_psd().plot(average = True, axes = ax, color = 'blue', ci = None)
raw_proj.compute_psd().plot(average = True, axes = ax, color = 'red', ci = None)

# %% Bad results from age prediction

data_file = os.path.join(deriv_root,"sub-0258/ses-02/meg/","sub-0258_ses-02_task-rest_proc-clean_epo.fif")
raw = mne.read_epochs(data_file)
raw.load_data()

raw_proj = raw.copy()

empty_room_file = os.path.join(deriv_root,"sub-0258/ses-02/meg/","sub-0258_ses-02_task-noise_proc-filt_raw.fif")
empty_room_raw = mne.io.read_raw_fif(empty_room_file)

empty_room_raw.del_proj()

empty_room_projs = mne.compute_proj_raw(empty_room_raw)

raw_proj.add_proj(empty_room_projs)
raw_proj.apply_proj()

fig = plt.figure(figsize = (10,5))
ax = plt.subplot(1,1,1)
fig.add_subplot(ax)

raw.compute_psd().plot(average = True, axes = ax, color = 'blue', ci = None)
raw_proj.compute_psd().plot(average = True, axes = ax, color = 'red', ci = None)

# %% Bad PSD 1

data_file = os.path.join(deriv_root,"sub-0378/ses-01/meg/","sub-0378_ses-01_task-rest_proc-clean_epo.fif")
raw = mne.read_epochs(data_file)
raw.load_data()

raw_proj = raw.copy()

empty_room_file = os.path.join(deriv_root,"sub-0378/ses-01/meg/","sub-0378_ses-01_task-noise_proc-filt_raw.fif")
empty_room_raw = mne.io.read_raw_fif(empty_room_file)

empty_room_raw.del_proj()

empty_room_projs = mne.compute_proj_raw(empty_room_raw)

raw_proj.add_proj(empty_room_projs)
raw_proj.apply_proj()

fig = plt.figure(figsize = (10,5))
ax = plt.subplot(1,1,1)
fig.add_subplot(ax)

raw.compute_psd().plot(average = True, axes = ax, color = 'blue', ci = None)
raw_proj.compute_psd().plot(average = True, axes = ax, color = 'red', ci = None)


# %% Bad PSD 2

data_file = os.path.join(deriv_root,"sub-0457/ses-01/meg/","sub-0457_ses-01_task-rest_proc-clean_epo.fif")
raw = mne.read_epochs(data_file)
raw.load_data()

raw_proj = raw.copy()

empty_room_file = os.path.join(deriv_root,"sub-0457/ses-01/meg/","sub-0457_ses-01_task-noise_proc-filt_raw.fif")
empty_room_raw = mne.io.read_raw_fif(empty_room_file)

empty_room_raw.del_proj()

empty_room_projs = mne.compute_proj_raw(empty_room_raw)

raw_proj.add_proj(empty_room_projs)
raw_proj.apply_proj()

fig = plt.figure(figsize = (10,5))
ax = plt.subplot(1,1,1)
fig.add_subplot(ax)

raw.compute_psd().plot(average = True, axes = ax, color = 'blue', ci = None)
raw_proj.compute_psd().plot(average = True, axes = ax, color = 'red', ci = None)


# %% Bad PSD 3

data_file = os.path.join(deriv_root,"sub-CONP0173/ses-02/meg/","sub-CONP0173_ses-02_task-rest_proc-clean_epo.fif")
raw = mne.read_epochs(data_file)
raw.load_data()

raw_proj = raw.copy()

empty_room_file = os.path.join(deriv_root,"sub-CONP0173/ses-02/meg/","sub-CONP0173_ses-02_task-noise_proc-filt_raw.fif")
empty_room_raw = mne.io.read_raw_fif(empty_room_file)

empty_room_raw.del_proj()

empty_room_projs = mne.compute_proj_raw(empty_room_raw)

raw_proj.add_proj(empty_room_projs)
raw_proj.apply_proj()

fig = plt.figure(figsize = (10,5))
ax = plt.subplot(1,1,1)
fig.add_subplot(ax)

raw.compute_psd().plot(average = True, axes = ax, color = 'blue', ci = None)
raw_proj.compute_psd().plot(average = True, axes = ax, color = 'red', ci = None)

# Autoreject
# 2, 1


# Compute features
# 1, 2

# Benchmark
# MAE(filterbank-riemann, omega) = 7.0246106695767185
# r2(filterbank-riemann, omega) = -0.3647212507252994

# dummy :
# MAE(dummy, omega) = 6.979008924949289
# r2(dummy, omega) = -0.07963111111175525


# %% PSD of worst subject

raw = mne.io.read_raw_ctf("/storage/store2/data/Omega/sub-CONP0173/ses-02/meg/sub-CONP0173_ses-02_task-rest_run-01_meg.ds")

raw.compute_psd().plot()

# %% SSP on some subjects
# All epochs, compute projectors on small frequencies

subjects = ["CONP0173"]
nb_proj = [2,5,10,15,20,30]
fig, axs = plt.subplots(nrows=7, ncols=1, figsize = (40,35))

for sub in subjects:
    session = subjects_data[subjects_data['subject_id']==sub]['session'].iloc[0]
    epoch_file = os.path.join(deriv_root,"sub-"+str(sub), "ses-0"+str(session), "meg","sub-"+str(sub)+"_ses-0"+str(session)+"_task-rest_proc-clean_epo.fif")
        
    epoch = mne.read_epochs(epoch_file)
    epoch_filtered = epoch.filter(0.1,7)

    epoch.compute_psd(picks = 'meg').plot(axes = axs.flat[0])
    axs.flat[0].set_title("No projector used")
    axs.flat[0].set_xlim([0,10])
    axs.flat[0].set_ylim([-10,120])
    
    for i, n_proj in enumerate(nb_proj):
        projectors = mne.compute_proj_epochs(epoch_filtered, n_mag=n_proj)
        epoch.add_proj(projectors)
        epoch.apply_proj()

        epoch.compute_psd(picks = 'meg').plot(axes = axs.flat[i+1])
        
        axs.flat[i+1].set_xlim([0,10])
        axs.flat[i+1].set_ylim([-10,120])
        axs.flat[i+1].set_title(str(n_proj )+" projectors used")

plt.show()


# %% PSD CONP0173 raw (0-70 HZ)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (15,5))

raw_path = os.path.join(bids_root, "sub-CONP0173/ses-02/meg/sub-CONP0173_ses-02_task-rest_run-01_meg.ds")
raw = mne.io.read_raw_ctf(raw_path)

raw.compute_psd(picks = 'meg').plot(axes = ax)

ax.set_xlim([0,70])

plt.show()

# %% Look at every preprocessing step

fig, axs = plt.subplots(nrows=4, ncols=1, figsize = (8,7))

raw_path = os.path.join(bids_root, "sub-CONP0173/ses-02/meg/sub-CONP0173_ses-02_task-rest_run-01_meg.ds")
raw = mne.io.read_raw_ctf(raw_path).load_data()
raw.compute_psd(picks = 'meg').plot(axes = axs.flat[0])
axs.flat[0].set_xlim([0,60])
axs.flat[0].set_ylim([-10,120])
axs.flat[0].set_title("Raw data")

raw_notch = raw.notch_filter(60)
raw_notch.compute_psd(picks = 'meg').plot(axes = axs.flat[1])
axs.flat[1].set_xlim([0,100])
axs.flat[1].set_ylim([-10,120])
axs.flat[1].set_title("Notch filter")

raw_filter = raw_notch.filter(0.1,49)
raw_filter.compute_psd(picks = 'meg').plot(axes = axs.flat[2])
axs.flat[2].set_xlim([0,60])
axs.flat[2].set_ylim([-10,120])
axs.flat[2].set_title("Notch filter + Bandpass filter")

raw_resample = raw_filter.resample(200)
raw_resample.compute_psd(picks = 'meg').plot(axes = axs.flat[3])
axs.flat[3].set_xlim([0,60])
axs.flat[3].set_ylim([-10,120])
axs.flat[3].set_title("Notch filter + Bandpass filter + Resample")

plt.show()

# %% Test of different freq for resample

fig, axs = plt.subplots(nrows=3, ncols=1, figsize = (40,35))

raw_path = os.path.join(bids_root, "sub-CONP0173/ses-02/meg/sub-CONP0173_ses-02_task-rest_run-01_meg.ds")
raw = mne.io.read_raw_ctf(raw_path).load_data()
raw.compute_psd(picks = 'meg').plot(axes = axs.flat[0])
axs.flat[0].set_xlim([0,100])
axs.flat[0].set_ylim([-10,120])
axs.flat[0].set_title("Raw data")


raw_resample = raw.resample(200)
raw_resample.compute_psd(picks = 'meg').plot(axes = axs.flat[1])
axs.flat[1].set_xlim([0,100])
axs.flat[1].set_ylim([-10,120])
axs.flat[1].set_title("Resample at 200Hz")


raw_resample = raw.resample(2000)
raw_resample.compute_psd(picks = 'meg').plot(axes = axs.flat[2])
axs.flat[2].set_xlim([0,100])
axs.flat[2].set_ylim([-10,120])
axs.flat[2].set_title("Resample at 2000Hz")

plt.show()

# %% Vérification filt_raw

path = '/storage/store3/work/kachardo/derivatives/omega/sub-0233/ses-01/meg/sub-0233_ses-01_task-rest_run-01_proc-filt_raw.fif'
raw_filt = mne.io.read_raw_fif(path)
raw_filt.info

# %% Plot avant et après preprocessing

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors = list(colors)
color = None

fig = plt.figure(figsize = (10,5))
ax_before = plt.subplot(2,1,1)
ax_after = plt.subplot(2,1,2)
fig.add_subplot(ax_before)
fig.add_subplot(ax_after)

for i, subject in enumerate(list(subjects_data[subjects_data['group'] == 'Control']['subject_id'])):
        session = subjects_data[subjects_data['subject_id']==subject]['session'].iloc[0]
        run = subjects_data[subjects_data['subject_id']==subject]['run'].iloc[0]
        
        epoch_before_path = os.path.join(bids_root,"sub-"+str(subject), "ses-0"+str(session), "meg","sub-"+str(subject)+"_ses-0"+str(session)+"_task-rest_run-0"+str(run)+"_meg.ds")
        epoch_after_path = os.path.join(deriv_root,"sub-"+str(subject), "ses-0"+str(session), "meg","sub-"+str(subject)+"_ses-0"+str(session)+"_task-rest_run-0"+str(run)+"_proc-filt_raw.fif")
        
        epoch_before = mne.io.read_raw_ctf(epoch_before_path)
        epoch_after = mne.io.read_raw_fif(epoch_after_path)
        
        if i >= len(colors):
            color = colors[i - len(colors)]
        else:
            color = colors[i]

        epoch_before.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax_before, ci = None)
        epoch_after.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax_after, ci = None)
            
ax_before.set_title('Average PSD for each Control subject (Raw)')
ax_before.set_xlim([0.1,50])
ax_before.set_ylim([0,100])
ax_after.set_title('Average PSD for each Control subject (Raw, after notch filter, bandpass filter and resampling)')
ax_after.set_xlim([0.1,50])
ax_after.set_ylim([0,100])

# %% Plot avant, pendant et après preprocessing (raw, avec notch et bandpass filter, epochs avec decim)

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors = list(colors)
color = None

fig = plt.figure(figsize = (10,5))
ax_before = plt.subplot(2,1,1)
ax_during = plt.subplot(2,1,1)
ax_after = plt.subplot(2,1,2)
fig.add_subplot(ax_before)
fig.add_subplot(ax_during)
fig.add_subplot(ax_after)

for i, subject in enumerate(list(subjects_data[subjects_data['group'] == 'Control']['subject_id'])):
        session = subjects_data[subjects_data['subject_id']==subject]['session'].iloc[0]
        run = subjects_data[subjects_data['subject_id']==subject]['run'].iloc[0]
        
        epoch_before_path = os.path.join(bids_root,"sub-"+str(subject), "ses-0"+str(session), "meg","sub-"+str(subject)+"_ses-0"+str(session)+"_task-rest_run-0"+str(run)+"_meg.ds")
        epoch_during_path = os.path.join(deriv_root,"sub-"+str(subject), "ses-0"+str(session), "meg","sub-"+str(subject)+"_ses-0"+str(session)+"_task-rest_run-0"+str(run)+"_proc-filt_raw.fif")
        epoch_after_path = os.path.join(deriv_root,"sub-"+str(subject), "ses-0"+str(session), "meg","sub-"+str(subject)+"_ses-0"+str(session)+"_task-rest_epo.fif")
        
        epoch_before = mne.io.read_raw_ctf(epoch_before_path)
        epoch_during = mne.io.read_raw_fif(epoch_during_path)
        epoch_after = mne.read_epochs(epoch_after_path)
        
        if i >= len(colors):
            color = colors[i - len(colors)]
        else:
            color = colors[i]

        epoch_before.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax_before, ci = None)
        epoch_during.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax_during, ci = None)
        epoch_after.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax_after, ci = None)
            
ax_before.set_title('Average PSD for each Control subject (Raw)')
ax_before.set_xlim([0.1,50])
ax_before.set_ylim([0,100])

ax_during.set_title('Average PSD for each Control subject (Raw, after notch filter and bandpass filter)')
ax_during.set_xlim([0.1,50])
ax_during.set_ylim([0,100])

ax_after.set_title('Average PSD for each Control subject (Epochs, after notch filter, bandpass filter and decim = 12)')
ax_after.set_xlim([0.1,50])
ax_after.set_ylim([0,100])

# %% Plot PSD after SSP on low frequencies for Control subjects
# Run mne-bids-pipeline with new ssp on all the subjects
# 23, 11, 12, 13, 21, 22

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

fig = plt.figure(figsize = (10,5))
ax = plt.subplot()
fig.add_subplot(ax)
i = 0
colors = list(colors)
color = None

for subject in os.listdir(deriv_root):
    if subject.startswith('sub'):
        id = subject[4:]
        if subjects_data[subjects_data['subject_id']==id]['group'].iloc[0] == 'Control':
            session = subjects_data[subjects_data['subject_id']==id]['session'].iloc[0]
            epoch_file = os.path.join(deriv_root,subject, "ses-0"+str(session), "meg","sub-"+str(id)+"_ses-0"+str(session)+"_task-rest_proc-clean_epo.fif")
            epoch = mne.read_epochs(epoch_file)
            if i >= len(colors):
                color = colors[i - len(colors)]
            else:
                color = colors[i]
            epoch.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax, ci = None)
            i += 1
            
ax.set_title('Average PSD for each Control subject')
plt.show()


# %% Plot PSD after autoreject

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

fig = plt.figure(figsize = (10,5))
ax = plt.subplot()
fig.add_subplot(ax)
i = 0
colors = list(colors)
color = None

for subject in os.listdir(deriv_root):
    if subject.startswith('sub'):
        id = subject[4:]
        if subjects_data[subjects_data['subject_id']==id]['group'].iloc[0] == 'Control':
            session = subjects_data[subjects_data['subject_id']==id]['session'].iloc[0]
            epoch_file = os.path.join(deriv_root,subject, "ses-0"+str(session), "meg","sub-"+str(id)+"_ses-0"+str(session)+"_task-rest_proc-autoreject_epo.fif")
            epoch = mne.read_epochs(epoch_file)
            if i >= len(colors):
                color = colors[i - len(colors)]
            else:
                color = colors[i]
            epoch.compute_psd(picks = 'meg').plot(color = color, picks = 'meg', average = True,axes = ax, ci = None)
            i += 1
            
ax.set_title('Average PSD for each Control subject after autoreject')
plt.show()

# %% Plot all PSDs

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
            epoch.compute_psd(picks = 'meg').plot(axes = axs.flat[i], picks = 'meg', average = True, ci = None)
            i +=1

plt.show()
fig.savefig('omega_subjects_control_psd_ssp_low.png')


# %% Plot distribution of covariances
from pyriemann.embedding import SpectralEmbedding
from coffeine import ProjCommonSpace

import mne
from mne import io

import matplotlib.pyplot as plt
# %% Open the covariances
features = h5io.read_hdf5(
            deriv_root / 'features_fb_covs_rest.h5')

# %% Alpha covariances of all control subjects
nb_control = subjects_data[subjects_data['group'] == 'Control'].shape[0]
covs_alpha = np.empty((nb_control, 262, 262))
i = 0

for sub in features :
    if subjects_data[subjects_data['subject_id'] == sub[4:]]['group'].iloc[0] == 'Control':
        covs_alpha[i] = features[sub]['covs'][0]
        i += 1

print(covs_alpha)

# %% Proj common space
pcs = ProjCommonSpace(n_compo=150)
covs_alpha_proj = pcs.fit(covs_alpha).transform(covs_alpha)
covs_alpha_proj = pd.DataFrame.to_numpy(covs_alpha_proj)
print(covs_alpha_proj)

# %% Remake array
covs_alpha_p = np.empty((nb_control, 150, 150))
for i in range(nb_control):
    covs_alpha_p[i] = covs_alpha_proj[i][0]
print(covs_alpha_p)

# %% Spectral embedding
se = SpectralEmbedding()
covs_embedding = se.fit(covs_alpha_p).fit_transform(covs_alpha_p)
print(covs_embedding)

# %% Age list
ages = []

for sub in list(features.keys()):
    if subjects_data[subjects_data['subject_id'] == sub[4:]]['group'].iloc[0] == 'Control':
        ages.append((all_subjects[all_subjects['participant_id'] == sub]['age'].iloc[0] / 7) ** 2)

ages

# %% Plot embeddings
plt.scatter(covs_embedding[:,0], covs_embedding[:,1], s = ages)
plt.show()