import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class AudioDataset(Dataset):
    """"""
    def __init__(self, h5_file, length_of_sample, num_angle_bins=None, distance_bins=None):
        """
        Args:
            h5_file: an hdf5 file (readable by h5py) that contains many groups.  Within each
    group is a timestamp (Nx1 float array), a gt_dist (Nx1 float array of ground truth distances in meters),
    and an array of audio samples from the microphone array (Nx8 float array)
            length_of_sample (int): the number of data points to include in each sample.  A sample will be a contiguous chunk
    of the arrays stored in the h5 file.  The length of this chunk is length_of_sample.
            num_angle_bins (int):  if present, convert angle labels into int labels corresponding to different
                angle bins.  If None then return floating point angles.
            distance_bins (list[int]): a list of cutoffs for the distance bins.  If you pass [100, 600, 1000, 1500, 2000]
                then the dist will be categories in bins that space [0-100], [100-600], etc.  If None, then return
                flaoting point distance measurements.
        """
        self.h5_file = h5_file
        self.keys = list(h5_file.keys())
        self.length_of_sample = length_of_sample
        self.num_angle_bins = num_angle_bins
        self.distance_bins = distance_bins


    def __len__(self):
       return len(self.keys)

    def __getitem__(self, idx):
        data = self.h5_file[self.keys[idx]]
        timestamp = data['timestamp']
        audio     = data['audio']
        dist      = data['dist']
        angle      = data['angle']
        los = self.length_of_sample

        if self.num_angle_bins is not None:
            interval = 2*np.pi/self.num_angle_bins
            ang_categories = np.zeros(angle.shape, dtype=np.int)
            for i in range(1, self.num_angle_bins):
                ang_categories[angle[:] > (-np.pi+i*interval)] += 1
            angle = ang_categories

        if self.distance_bins is not None:
            dist_categories = np.zeros(dist.shape, dtype=np.int)
            for db in self.distance_bins:
                dist_categories[dist[:] > db] += 1
            dist = dist_categories

        start = np.random.randint(0, len(timestamp)-los )
        return timestamp[start:start+los], audio[start:start+los], dist[start:start+los], angle[start:start+los]




# f = h5py.File('temp.h5','w')
# grp = f.create_group('test_group')
# a = np.linspace(0,100,101)
# grp['timestamp'] = np.linspace(0,100,101)
# grp['audio'] = np.linspace(0,1,101)
# grp['dist'] = np.linspace(0,2000,101)
# grp['angle'] = np.linspace(-np.pi,np.pi,101)
# f.close()

f = h5py.File('temp.h5','r')
ds = AudioDataset(f, length_of_sample=99, num_angle_bins=10, distance_bins=[600,1000,1500])

ts, audio, dist, angle = ds[0]
print('timestamp: ', ts)
print('audio: ', audio)
print('dist: ', dist)
print('angle: ', angle)