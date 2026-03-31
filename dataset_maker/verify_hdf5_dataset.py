# HDF5 STRUCTURE (simplified)
# dataset
#  ├── sub-001_task-eyesclosed_eeg
#  │    └── eeg  (dataset: shape = (19, 117960))
#  ├── sub-002_task-eyesclosed_eeg
#  │    └── eeg
#  ├── sub-003_task-eyesclosed_eeg
#  │    └── eeg
#  ...
#  └── sub-088_task-eyesclosed_eeg
#       └── eeg

# WHAT THE DATA MEANS

# dataset["sub-001_task-eyesclosed_eeg"]['eeg'].shape = (19, 117960)

# 19:
#   - Number of EEG channels
#   - Each row = one channel (e.g., Fp1, Fp2, C3, etc.)

# 117960:
#   - Number of time samples per channel
#   - Each column = signal value at a specific time point
#   - Resampled to 200 Hz in make_h5dataset_for_pretrain.py
#   - Total recording length is 117960 / 200 = 589.8 seconds (~9.8 minutes)

# data looks like:
# [[10.09, 29.30, 21.72, ..., 24.75, 13.73, 17.03],
#  [ 0.09, 14.31, 18.22, ..., 63.26, 40.98, 50.84],
#  ...
# ]

# INTERPRETATION:
# - data[channel_index, time_index]
# - Each number = EEG signal amplitude at that moment
# - Units are in microvolts (µV)

# Example:
# - data[0, :] → full time series for channel 0
# - data[:, 1000] → values from all 19 channels at time step 1000

# In short:
# - Rows = different brain sensors
# - Columns = signal over time
# - Values = brain electrical activity measurements

import h5py

dataset = h5py.File("../output/dataset.hdf5", "r")

# print(list(dataset.keys()))

sub = dataset["sub-001_task-eyesclosed_eeg"]
# print(list(sub.keys()))

print(sub['eeg'])

data = sub["eeg"][:]   # loads into a NumPy array
print(data.shape)
print(data[:5])        # preview

dataset.close()
