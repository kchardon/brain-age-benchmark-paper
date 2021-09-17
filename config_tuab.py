import pathlib
import mne

study_name = "age-prediction-benchmark"

bids_root = pathlib.Path(
    "/storage/store2/data/TUAB-healthy-bids-nosplit")

deriv_root = pathlib.Path(
    "/storage/store3/derivatives/TUAB-healthy-bids-nosplit")
# "/storage/store2/derivatives/eeg-pred-modeling-summer-school/")

task = "rest"

sessions = ["001"]

datatype = "eeg"
ch_types = ["eeg"]

l_freq = 0.1
h_freq = 49

eeg_reference = []

find_breaks = False

spatial_filter = None

reject = None

on_error = "abort"
on_rename_missing_events = "warn"

N_JOBS = 30

epochs_tmin = 0
epochs_tmax = 10
baseline = None

event_repeated = "drop"
l_trans_bandwidth = "auto"

h_trans_bandwidth = "auto"

random_state = 42

shortest_event = 1

log_level = "info"

mne_log_level = "info"
# on_error = 'continue'
# on_error = "continue"

on_error = 'abort'
