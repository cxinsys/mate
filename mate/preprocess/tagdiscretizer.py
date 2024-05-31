import numpy as np

from mate.preprocess.discretizer import Discretizer

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