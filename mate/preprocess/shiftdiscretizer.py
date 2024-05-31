import numpy as np

from mate.preprocess.discretizer import Discretizer

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
            raise ValueError("method should be designated: %s"%(mode))



        return arrs, n_bins