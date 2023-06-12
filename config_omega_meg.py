import pathlib
import os


bids_root = pathlib.Path('/storage/store2/data/Omega')
deriv_root = pathlib.Path('/storage/store3/work/kachardo/derivatives/omega')

subjects_dir = pathlib.Path('/storage/store/data/camcan-mne/freesurfer')

subjects = ['0523']

process_empty_room = True

drop_channels = [
    'MLC63', 'MLO52', 'MLT26', 'MLT55',
    'MRP52', 'MRT51', 'CZ', 'CP3',
    'CP4', 'EEG00', 'MLC25', 'MLC62', 'MLO53',
    'MLP55', 'MRF64', 'MRP53', 'EEG01',
    'EEG02', 'EEG05', 'EEG06', 'EEG03',
    'EEG04', 'EEG004']

analyze_channels = ['ECG', 'VEOG',
'HEOG','SCLK01','BG1','BG2','BG3','BP1','BP2','BP3','BR1','BR2','BR3','G11',
 'G12','G13','G22','G23','P11','P12','P13','P22','P23','Q11','Q12','Q22','Q23','R12','R13','R22',
 'MLC11','MLC12','MLC13','MLC14','MLC15','MLC16','MLC17','MLC21', 'MLC22', 'MLC23', 'MLC24', 'MLC31',
 'MLC32', 'MLC41', 'MLC42', 'MLC51', 'MLC52', 'MLC53', 'MLC54', 'MLC55', 'MLC61', 'MLF11', 'MLF12', 'MLF13',
 'MLF14', 'MLF21', 'MLF22', 'MLF23', 'MLF24', 'MLF25', 'MLF31', 'MLF32', 'MLF33', 'MLF34', 'MLF35', 'MLF41',
 'MLF42', 'MLF43', 'MLF44', 'MLF45', 'MLF46', 'MLF51', 'MLF52', 'MLF53', 'MLF54', 'MLF55', 'MLF56', 'MLF61',
 'MLF62', 'MLF63', 'MLF64', 'MLF65', 'MLF66', 'MLF67', 'MLO11', 'MLO12', 'MLO13', 'MLO14', 'MLO21', 'MLO22',
 'MLO23', 'MLO24', 'MLO31', 'MLO32', 'MLO33', 'MLO34', 'MLO41', 'MLO42', 'MLO43', 'MLO44', 'MLO51', 'MLP11',
 'MLP12', 'MLP21', 'MLP22', 'MLP23', 'MLP31', 'MLP32', 'MLP33', 'MLP34', 'MLP35', 'MLP41', 'MLP42', 'MLP43',
 'MLP44', 'MLP45', 'MLP51', 'MLP52', 'MLP53', 'MLP54', 'MLP56', 'MLP57', 'MLT11', 'MLT12', 'MLT13', 'MLT14',
 'MLT15', 'MLT16', 'MLT21', 'MLT22', 'MLT23', 'MLT24', 'MLT25', 'MLT27', 'MLT31', 'MLT32', 'MLT33', 'MLT34',
 'MLT35', 'MLT36', 'MLT37', 'MLT41', 'MLT42', 'MLT43', 'MLT44', 'MLT45', 'MLT46', 'MLT47', 'MLT51', 'MLT52',
 'MLT53', 'MLT54', 'MLT56', 'MLT57', 'MRC11', 'MRC12', 'MRC13', 'MRC14', 'MRC15', 'MRC16', 'MRC17', 'MRC21',
 'MRC22', 'MRC23', 'MRC24', 'MRC25', 'MRC31', 'MRC32', 'MRC41', 'MRC42', 'MRC51', 'MRC52', 'MRC53', 'MRC54',
 'MRC55', 'MRC61', 'MRC62', 'MRC63', 'MRF11', 'MRF12', 'MRF13', 'MRF14', 'MRF21', 'MRF22', 'MRF23', 'MRF24', 'MRF25',
 'MRF31', 'MRF32', 'MRF33', 'MRF34', 'MRF35', 'MRF41', 'MRF42', 'MRF43', 'MRF44', 'MRF45', 'MRF46', 'MRF51',
 'MRF52', 'MRF53', 'MRF54', 'MRF55', 'MRF56', 'MRF61', 'MRF62', 'MRF63', 'MRF65', 'MRF66', 'MRF67', 'MRO11', 'MRO12',
 'MRO13', 'MRO14', 'MRO21', 'MRO22', 'MRO23', 'MRO24', 'MRO31', 'MRO32', 'MRO33', 'MRO34', 'MRO41', 'MRO42',
 'MRO43', 'MRO44', 'MRO51', 'MRO52', 'MRO53', 'MRP11', 'MRP12', 'MRP21', 'MRP22', 'MRP23', 'MRP31', 'MRP32', 'MRP33',
 'MRP34', 'MRP35', 'MRP41', 'MRP42', 'MRP43', 'MRP44', 'MRP45', 'MRP51', 'MRP54', 'MRP55', 'MRP56', 'MRP57',
 'MRT11', 'MRT12', 'MRT13', 'MRT14', 'MRT15', 'MRT16', 'MRT21', 'MRT22', 'MRT23', 'MRT24', 'MRT25', 'MRT26',
 'MRT31', 'MRT32', 'MRT33', 'MRT34', 'MRT35', 'MRT36', 'MRT37', 'MRT41', 'MRT42', 'MRT43', 'MRT44', 'MRT45',
 'MRT46', 'MRT47', 'MRT52', 'MRT53', 'MRT54', 'MRT55', 'MRT56', 'MRT57', 'MZC01', 'MZC02', 'MZC03', 'MZC04',
 'MZF01', 'MZF02', 'MZF03', 'MZO01', 'MZO02', 'MZO03', 'MZP01', 'HADC001', 'HADC002', 'HADC003', 'HDAC001',
 'HDAC002', 'HDAC003', 'HLC0011', 'HLC0012', 'HLC0013', 'HLC0021', 'HLC0022', 'HLC0023', 'HLC0031', 'HLC0032',
 'HLC0033', 'HLC0018', 'HLC0028', 'HLC0038', 'HLC0014', 'HLC0015', 'HLC0016', 'HLC0017', 'HLC0024', 'HLC0025',
 'HLC0026', 'HLC0027', 'HLC0034', 'HLC0035', 'HLC0036', 'HLC0037']

source_info_path_update = {'processing': 'autoreject',
                           'suffix': 'epo'}

task_is_rest = True
task = 'rest'
sessions = ['02']
runs = ['03']
data_type = 'meg'
ch_types = ['meg']

l_freq = 0.1
h_freq = 49
notch_freq = 60


eog_channels = ["VEOG", "HEOG"]

spatial_filter = 'ssp'
#n_proj_eog = 1

reject = None

raw_resample_sfreq = 200

epochs_tmin = 0.
epochs_tmax = 10.
rest_epochs_overlap = 0.
rest_epochs_duration = 10.

baseline = None

n_jobs = 40
on_error = 'continue'
log_level = 'info'
