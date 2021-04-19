"""Download datasets with recorded audio from Zips while labeled intruders are present.
This audio will be written into an h5 file.  The file contains groups, one per flight.  For
each flight, the group contains a timestamp (in seconds), audio samples, and group truth labels. The
ground truth data in interpolated so it shares a timestamp with the audio samples.  This makes it
easy to create lablled datasets.
"""

from pprint import pprint
import argparse
import numpy as np
import pandas as pd
idx = pd.IndexSlice
import seaborn as sns
import matplotlib.pyplot as plt
from air_daa_offline.database import dataset_generation
from air_daa_offline.database import populate_datastore
from air_daa_offline import Tasks
import h5py

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start', type=int, help='first entry to grab')
parser.add_argument('--end', type=int, help='last entry to grab')
args = parser.parse_args()


print(f'Creating dataset object: {args.star} - {args.end}')
## Create a data store and populate it with all the data.  This can take a long time.
sys_params = populate_datastore.get_default_sys_params()
#encs = dataset_generation.get_train_dataset(sys_params, nsample=None, index_slice=slice(args.start,args.end))
encs = dataset_generation.get_dev_dataset(sys_params, nsample=None, index_slice=slice(args.start,args.end))


train_name = "tom_train_store.h5"

print('Populate datastore...')
store = populate_datastore.EvalDatastore(train_name)
print('Populate meta data...')
populate_datastore.populate_sc_meta(encs, store)
print('Populate ground truth...')
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
    #print('   processing id: ', logs_id)
    models_out = helper.get_models_out(logs_id, sys_params)
    #print('   processing audio')
    audio_arr = models_out.enc_sources.ess.audio_arr
    #print('   audio shape: ', audio_arr.shape)
    time_arr = models_out.enc_sources.ess.audio_utcdatetime
    audio_min = time_arr.min()
    audio_max = time_arr.max()
    assert len(audio_arr) == len(time_arr), 'Audio and timestamp have incompatible dimensions'
    assert audio_arr.shape[1] == 8, 'Number of mics in audio source is not 8'

    # Get the ground truth distance data (meters).
    gt_dist = df_gt['gt_r_b'][logs_id]
    gt_min = gt_dist.index.min()
    gt_max = gt_dist.index.max()

    assert gt_min <= audio_min, f"[{logs_id}]: GT record starts after audio.  Cannot interpolate GT onto audio timestamp."
    assert gt_max >= audio_max, f"[{logs_id}]: GT record ends before audio.  Cannot interpolate GT onto audio timestamp."

    # The GT timestamp is not the same as the audio timestamp, so do interpolation.
    s = pd.Series(index=time_arr, dtype=np.float64)
    gt_dist_interp = gt_dist.append(s).interpolate(method='cubic')
    gt_dist_interp = gt_dist_interp[s.index]

    # Get the ground truth angle data (meters).
    gt_dist = df_gt['gt_az_b'][logs_id]
    s = pd.Series(index=time_arr, dtype=np.float64)
    gt_angle_interp = gt_dist.append(s).interpolate(method='cubic')
    gt_angle_interp = gt_angle_interp[s.index]

    # Plot results to show that interpolated GT looks like original GT, but with different points in the timestamp.
    #     plt.subplots(figsize=(18,8 ))
    #     pyplot.subplot(1,2,1)
    #     gt_vals.plot()
    #     pyplot.subplot(1,2,2)
    #     gt_interp.plot()

    # Return nx1 timestamp, nx8 audio, nx1 ground truth
    return time_arr, audio_arr, gt_dist_interp.to_numpy(), gt_angle_interp.to_numpy()


## Throw out the datasets with no ground truth, and
gt_valid = df_gt.dropna()  # drop the data samples with no GT labels

## Create the h5 file and write to it
dataset_home = '/optional_dirs/datasets/raw_audio_and_groundtruth.h5'
print('Creating h5 file at location ', dataset_home)
f = h5py.File(dataset_home, 'a')

print('Begin processing flights ', dataset_home)

flight_names = set(i[0] for i in gt_valid.index)

for n in flight_names:  # Loop over flight names
    print('    Processing ', n)
    if n not in f.keys():
        try:
            time_arr, audio_arr, gt_dist, gt_angle = create_interpolated_ground_truth(n)
            grp = f.create_group(n)
            print('   numel = ', audio_arr.size)
            grp['timestamp'] = time_arr
            grp['audio'] = audio_arr
            grp['dist'] = gt_dist
            grp['angle'] = gt_angle
            f.flush()
        except Exception as e:
            print(str(e))

f.close()