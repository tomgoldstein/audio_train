"""Download datasets with recorded audio from Zips while labeled intruders are present.
This audio will be written into an h5 file.  The file contains groups, one per flight.  For
each flight, the group contains a timestamp (in seconds), audio samples, and group truth labels. The
ground truth data in interpolated so it shares a timestamp with the audio samples.  This makes it
easy to create lablled datasets.
"""

from pprint import pprint
import numpy as np
import pandas as pd
idx = pd.IndexSlice
import seaborn as sns
import matplotlib.pyplot as plt
from air_daa_offline.database import dataset_generation
from air_daa_offline.database import populate_datastore
from air_daa_offline import Tasks
import h5py

## Create a data store and populate it with all the data.  This can take a long time.
sys_params = populate_datastore.get_default_sys_params()
encs = dataset_generation.get_train_dataset(sys_params, nsample=None)

train_name = "tom_train_store.h5"

store = populate_datastore.EvalDatastore(train_name)
populate_datastore.populate_sc_meta(encs, store)
populate_datastore.populate_ground_truth(encs, store)

df_sc_meta = store.load('SC_META')
df_gt = store.load('GroundTruth')

#populate_datastore.populate_exclusions(encs, df_sc_meta, store)

helper = Tasks.DAGBackedModelsOutHelper()

def create_interpolated_ground_truth(logs_id):
    """Interpolate the GT data into the same timestamp as the audio from the flight.
    Also check for data validity: the GT timestamp needs to cover a wider temporal range than the audio timestamp.
    If not, the GT interpolation will produce invalid results"""
    # Extract audio data, and it's timestamp
    models_out = helper.get_models_out(logs_id, sys_params)
    audio_arr = models_out.enc_sources.ess.audio_arr
    time_arr = models_out.enc_sources.ess.audio_utcdatetime
    audio_min = time_arr.min()
    audio_max = time_arr.max()
    assert len(audio_arr) == len(time_arr), 'Audio and timestamp have incompatible dimensions'
    assert audio_arr.shape[1] == 8, 'Number of mics in audio source is not 8'

    # Get the ground truth data.
    gt_vals = df_gt['gt_r_b'][logs_id]
    gt_min = gt_vals.index.min()
    gt_max = gt_vals.index.max()

    assert gt_min <= audio_min, f"[{logs_id}]: GT record starts after audio.  Cannot interpolate GT onto audio timestamp."
    assert gt_max >= audio_max, f"[{logs_id}]: GT record ends before audio.  Cannot interpolate GT onto audio timestamp."

    # The GT timestamp is not the same as the audio timestamp, so do interpolation.
    s = pd.Series(index=time_arr, dtype=np.float64)
    gt_interp = gt_vals.append(s).interpolate(method='cubic')
    gt_interp = gt_interp[s.index]

    # Plot results to show that interpolated GT looks like original GT, but with different points in the timestamp.
    #     plt.subplots(figsize=(18,8 ))
    #     pyplot.subplot(1,2,1)
    #     gt_vals.plot()
    #     pyplot.subplot(1,2,2)
    #     gt_interp.plot()

    # Return nx1 timestamp, nx8 audio, nx1 ground truth
    return time_arr, audio_arr, gt_interp.to_numpy()


## Throw out the datasets with no ground truth, and
gt_valid = df_gt.dropna()  # drop the data samples with no GT labels

## Create the h5 file and write to it
dataset_home = '/home/ubuntu/datasets/raw_audio_and_groundtruth.h5'
f = h5py.File(dataset_home, 'w')

for i in gt_valid.index:  # Loop over flight names
    print('Processing ', i[0])
    time_arr, audio_arr, gt_interp = create_interpolated_ground_truth(i[0])
    grp = f.create_group(i[0])
    grp['timestamp'] = time_arr
    grp['audio'] = audio_arr
    grp['ground_truth'] = gt_interp

f.close()