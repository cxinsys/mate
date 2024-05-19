import os
import os.path as osp
import gc
import glob
import random
from itertools import permutations

import numpy as np
from scipy.signal import savgol_filter

import torch
from torch.utils.data import Dataset, DataLoader

class PairDataSet(Dataset):
    def __init__(self,
                 arr=None,
                 pairs=None):
        self._arr = arr

        if pairs is None:
            self._pairs = self.create_pairs(arr=arr)
        else:
            self._pairs = pairs

    def create_pairs(self, arr):
        pairs = permutations(range(len(arr)), 2)
        return np.asarray(tuple(pairs), dtype=np.int32)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, item):
        # return torch.Tensor(self._pairs[item])
        return self._pairs[item]
        # return self._arr, self._pairs[item]