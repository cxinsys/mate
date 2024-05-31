import numpy as np

from mate.preprocess.discretizer import Discretizer

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