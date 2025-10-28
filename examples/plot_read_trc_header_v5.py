from pathlib import Path
import numpy as np
from mne import create_info
from mne.io import RawArray, read_raw_edf
from micromed_io.trc import MicromedTRC
from neo.io import MicromedIO


## Load EDF file (mne) ##
## ------------------- ##

edf_fpath = Path("E:/EEG_1.edf")
raw_from_edf_mne = read_raw_edf(edf_fpath, preload=True)


## Load TRC file (micromedio) ##
## -------------------------- ##

trc_fpath = Path("E:/EEG_1.trc")
trc_data_mmio = MicromedTRC(trc_fpath)

# Unpack data
signals = trc_data_mmio.get_data(use_volt=True)
ch_names = trc_data_mmio.get_header().ch_names
fs = trc_data_mmio.get_sfreq()
labels = trc_data_mmio.get_markers()

# Create MNE Raw object
info = create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
raw_from_trc_mmio = RawArray(signals, info)

# # From microvolts to volts
# unit_factor = 1e-6  # uV
# raw_from_trc_neo.apply_function(lambda x: x * unit_factor)



## Load TRC file (python-neo) ##
## -------------------------- ##

trc_fpath = Path("E:/EEG_1.trc")
trc_data_neo = MicromedIO(trc_fpath)

# Unpack data
signals = np.array(trc_data_neo.read_segment().analogsignals[0]).transpose()
ch_names = np.array(trc_data_neo.header['signal_channels'], dtype="U")
fs = trc_data_neo._sampling_rate
labels = trc_data_neo._raw_events[0][0]

# format labels (per sample)
labels_ = np.zeros(signals.shape[-1], dtype='int') 
labels_[labels['start']] = labels['label']
labels = labels_

# Create MNE Raw object
info = create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
raw_from_trc_neo = RawArray(signals, info)



## Inspect loaded data ##
## ------------------- ##

raw_plot = raw_from_trc_mmio.copy().crop(tmin=0, tmax=10)

# Drop channels (misc and eeg)
misc_channels = ['ECG+-ECG-', 'MKR 1+-MKR 1-', 'MKR 2+-MKR 2-']
eeg_channels = ch_names[:28]
raw_plot.drop_channels(misc_channels + eeg_channels)

# Re-reference (CAR)
raw_plot.set_eeg_reference('average')

# Plot picks
picks = None # all channels
picks = ch_names[30:76] # all ECOG
picks = ['LID1-G2', 'LG1-G2', 'LS1-G2', 'LCD1-G2'] 

# Plot scaling
scalings = {'eeg': 100e-6}  

# Plot
_ = raw_plot.plot(picks=picks, scalings=scalings)



## Compare data ##
## ------------ ##

raw1 = raw_from_trc_mmio
raw2 = raw_from_edf_mne

print(f"Data shape raw1: {raw1.get_data().shape}")
print(f"Data shape raw2: {raw2.get_data().shape}")
