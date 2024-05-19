import time

import numpy as np
from scipy.signal import savgol_filter
import lightning.pytorch as pl
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader

from mate.transferentropy import TELightning
from mate import MATE
from mate.dataset import PairDataSet

# try:
#     from .mate.models.layer import LightningTE
#     from .mate.dataset.dataset import PairDataSet
# except (ImportError, ModuleNotFoundError) as err:
#     from mate.models.layer import LightningTE
#     from mate.dataset.dataset import PairDataSet

class MATELightning(MATE):
    def __init__(self,
                 arr=None,
                 pairs=None,
                 kp=0.5,
                 num_kernels=1,
                 method='pushing',
                 percentile=0,
                 smooth_func=None,
                 smooth_param=None,
                 len_time=None,
                 dt=1):
        super().__init__()

        self._pairs = pairs
        self._arr = arr
        self._bin_arr, self._n_bins = self.create_kde_array(kp=kp,
                                                              num_kernels=num_kernels,
                                                              method=method)
        self._devices = None

        self.model = TELightning(arr=self._bin_arr, len_time=len_time, dt=dt, n_bins=self._n_bins)
        self.dset_pair = PairDataSet(arr=self._bin_arr, pairs=self._pairs)

    def kernel_width(self, arr, kp=None, percentile=None):
        if percentile > 0:
            arr.sort(axis=1)
            i_beg = int(arr.shape[1] / 100 * percentile)
            std = np.std(arr[:, i_beg:-i_beg], axis=1, ddof=1)
        else:
            std = np.std(arr, axis=1, ddof=1)
        kw = kp * std
        kw[kw == 0] = 1

        return kw

    def custom_collate(self, batch):
        n_devices = None

        if type(self._devices)==int:
            n_devices = self._devices
        elif type(self._devices)==list:
            n_devices = len(self._devices)

        pairs = [item for item in batch]

        # arr = batch[0][0]

        return np.stack(pairs)

    def run(self,
            device=None,
            devices=None,
            batch_size=None,
            num_workers=0):

        self._devices = devices

        dloader_pair = DataLoader(self.dset_pair,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=self.custom_collate)

        trainer = L.Trainer(accelerator=device,
                            devices=devices,
                            num_nodes=1,
                            strategy="auto")


        trainer.predict(self.model, dloader_pair)

        if trainer.is_global_zero:
            results = self.model.return_result()

            return results


