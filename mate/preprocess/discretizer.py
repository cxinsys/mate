import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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

        n_bins = np.ceil((maxs - mins) / (self._kp * stds)).T.astype(self._dtype)

        bin_arr = np.floor((arr.T - mins) / (self._kp * stds)).T.astype(self._dtype)
        arrs = bin_arr[..., None]

        return arrs, n_bins

class InterpDiscretizer(Discretizer):
    def __init__(self,
                 # num_kernels=1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def binning(self, arr):
        arrs = []

        stds = np.std(arr, axis=1, ddof=1)
        mins = np.min(arr, axis=1)
        maxs = np.max(arr, axis=1)

        n_bins = np.ceil((maxs - mins) / stds).T.astype(self._dtype)

        bin_arr = ((arr.T - mins) / stds).T
        mid_arr = (bin_arr[:, :-1] + bin_arr[:, 1:]) / 2

        inter_arr = np.zeros((len(bin_arr), len(bin_arr[0]) + len(mid_arr[0])))

        inter_arr[:, ::2] = bin_arr
        inter_arr[:, 1::2] = mid_arr

        # Int Bin
        # inter_arr = np.floor(inter_arr).astype(dtype)

        # Float Bin
        inter_arr = inter_arr.astype(np.float32)

        # inter_arr = np.where(inter_arr < 0, 0, inter_arr)
        # inter_arr = np.where(inter_arr >= n_bins.reshape(-1, 1), (n_bins - 1).reshape(-1, 1), inter_arr)

        arrs = inter_arr[..., None]


        return arrs, n_bins

class ShiftDiscretizer(Discretizer):
    def __init__(self,
                 # num_kernels=1,
                 method,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._method = method

    def binning(self, arr):
        arrs = []

        stds = np.std(arr, axis=1, ddof=1)
        mins = np.min(arr, axis=1)
        maxs = np.max(arr, axis=1)

        n_bins = np.ceil((maxs - mins) / stds).T.astype(self._dtype)

        if 'shift_left' in self._method:
            bin_arr = np.floor((arr.T - (mins - (self._kp * stds))) / stds).T.astype(self._dtype)
            arrs = bin_arr[..., None]

        elif 'shift_right' in self._method:
            bin_arr = np.floor((arr.T - (mins + (self._kp * stds))) / stds).T.astype(self._dtype)

            arrs = bin_arr[..., None]

        elif 'shift_both' in self._method:
            for i in range(3):
                if i % 2 == 1:  # odd
                    bin_arr = np.floor((arr.T - (mins + ((i // 2 + i % 2) * self._kp * stds))) / stds).T.astype(self._dtype)  # pull
                else:
                    bin_arr = np.floor((arr.T - (mins - (i // 2 * self._kp * stds))) / stds).T.astype(self._dtype)  # push

                bin_arr = bin_arr.astype(self._dtype)

                arrs.append(bin_arr)
            arrs = np.stack(arrs, axis=2)

        else:
            raise ValueError("method should be designated: %s"%(self._method))



        return arrs, n_bins

class TagDiscretizer(Discretizer):
    def __init__(self,
                 # num_kernels=1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def binning(self, arr):
        arrs = []

        stds = np.std(arr, axis=1, ddof=1)
        mins = np.min(arr, axis=1)
        maxs = np.max(arr, axis=1)

        n_bins = np.ceil((maxs - mins) / stds).T.astype(self._dtype)

        for i in range(3):
            if i % 2 == 1:  # odd
                bin_arr = np.floor((arr.T - (mins + ((i // 2 + i % 2) * self._kp * stds))) / stds).T.astype(self._dtype)
            else:
                bin_arr = np.floor((arr.T - (mins - (i // 2 * self._kp * stds))) / stds).T.astype(self._dtype)

            # bin_arr = np.where(bin_arr < 0, 0, bin_arr)
            # bin_arr = np.where(bin_arr >= n_bins.reshape(-1, 1), (n_bins - 1).reshape(-1, 1), bin_arr)

            bin_maxs = np.max(bin_arr, axis=1)

            coeff = (i + 1) * 10 ** np.ceil(np.log10(bin_maxs))

            bin_arr += coeff[..., None].astype(self._dtype)

            arrs.append(bin_arr)

        arrs = np.stack(arrs, axis=2)

        return arrs, n_bins

class FixedWidthDiscretizer(Discretizer):
    def __init__(self, family, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = 10

        if family:
            if 'num_bins' in family:
                self.n_bins = family['num_bins']

    def binning(self, arr):
        binned_data = []

        for data in arr:
            bins = np.linspace(np.min(data), np.max(data), self.n_bins)
            tmp_data = np.digitize(data, bins)
            binned_data.append(tmp_data)

        arrs = np.array(binned_data)
        return np.expand_dims(arrs, -1), self.n_bins

class QuantileDiscretizer(Discretizer):
    def __init__(self, family, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = 5

        if family:
            if 'num_bins' in family:
                self.n_bins = family['num_bins']

    def binning(self, arr):
        binned_data = []

        for data in arr:
            tmp_data, bins = pd.qcut(data, self.n_bins, retbins=True, duplicates='drop')
            labels = np.arange(len(bins) - 1)
            tmp_data = pd.qcut(data, self.n_bins, labels, duplicates='drop')
            binned_data.append(list(tmp_data))

        arrs = np.array(binned_data)
        return np.expand_dims(arrs, -1), self.n_bins

class KmeansDiscretizer(Discretizer):
    def __init__(self, family, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = 5

        if family:
            if 'num_bins' in family:
                self.n_bins = family['num_bins']

    def binning(self, arr):
        binned_data = []

        for data in arr:
            kmeans = KMeans(n_clusters=self.n_bins, random_state=0).fit(data.reshape(-1, 1))
            tmp_data = list(kmeans.labels_)
            binned_data.append(tmp_data)

        arrs = np.array(binned_data)
        return np.expand_dims(arrs, -1), self.n_bins

class LogDiscretizer(Discretizer):
    def __init__(self, family, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = 10

        if family:
            if 'num_bins' in family:
                self.n_bins = family['num_bins']

    def binning(self, arr):
        binned_data = []

        for data in arr:
            logmax = np.ceil(np.max(np.log10(data))).astype(int)
            logmin = np.floor(np.min(np.log10(data))).astype(int)
            log_bins = np.logspace(logmin, logmax, self.n_bins)
            tmp_data = np.digitize(data, log_bins)
            binned_data.append(tmp_data)

        arrs = np.array(binned_data)
        return np.expand_dims(arrs, -1), self.n_bins