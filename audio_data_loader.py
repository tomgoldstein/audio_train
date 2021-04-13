import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class AudioDataset(Dataset):
    """h5_file: an hdf5 file (readable by h5py) that contains many groups.  Within each
    group is a timestamp (Nx1 float array), a gt_dist (Nx1 float array of ground truth distances in meters),
    and an array of audio samples from the microphone array (Nx8 float array)
    length_of_sample: the number of data points to include in each sample.  A sample will be a contiguous chunk
    of the arrays stored in the h5 file.  The length of this chunk is length_of_sample."""
    def __init__(self, h5_file, length_of_sample):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.h5_file = h5_file
        self.keys = list(h5_file.keys())
        self.length_of_sample = length_of_sample


    def __len__(self):
       return len(self.keys)

    def __getitem__(self, idx):
        data = self.h5_file[self.keys[idx]]
        timestamp = data['timestamp']
        audio     = data['audio']
        dist      = data['dist']
        angle      = data['angle']
        los = self.length_of_sample

        start = np.random.randint(0, len(timestamp)-los )
        return timestamp[start:start+los], audio[start:start+los], dist[start:start+los], angle[start:start+los]


# f = h5py.File('temp.h5','w')
# grp = f.create_group('test_group')
# a = np.linspace(0,100,101)
# grp['timestamp'] = a
# grp['audio'] = a+0.1
# grp['dist'] = a+0.2
# grp['angle'] = a+0.3
# f.close()

f = h5py.File('temp.h5','r')
ds = AudioDataset(f, length_of_sample=10)

ts, audio, dist, angle = ds[0]
print('timestamp: ', ts)
print('audio: ', audio)
print('dist: ', dist)
print('angle: ', angle)