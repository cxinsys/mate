import math

import numpy as np

class Discretizer():
    def __init__(self,
                 kp=0.5,
                 # percentile=0,
                 dtype=np.int32
                 ):

        self._kp = kp
        # self._percentile = percentile
        self._dtype = dtype

    def binning(self, arr):

        stds = np.std(arr, axis=1, ddof=1)
        mins = np.min(arr, axis=1)
        maxs = np.max(arr, axis=1)

        n_bins = np.ceil((maxs - mins) / stds).T.astype(self._dtype)

        bin_arr = np.floor((arr.T - mins) / (self._kp * stds)).T.astype(self._dtype)
        arrs = bin_arr[..., None]

        return arrs, n_bins