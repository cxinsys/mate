import time

import numpy as np
from scipy.signal import savgol_filter
import lightning.pytorch as pl
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader

from mate.transferentropy import TELightning
from mate.dataset import PairDataSet

# try:
#     from .mate.models.layer import LightningTE
#     from .mate.dataset.dataset import PairDataSet
# except (ImportError, ModuleNotFoundError) as err:
#     from mate.models.layer import LightningTE
#     from mate.dataset.dataset import PairDataSet

class MATELightning(object):
    def __init__(self,
                 arr=None,
                 pairs=None,
                 kp=0.5,
                 percentile=0,
                 win_length=10,
                 polyorder=3,
                 len_time=None,
                 dt=1):
        super().__init__()

        self._pairs = pairs

        self._bin_arr = self.create_binned_array(arr=arr,
                                            kp=kp,
                                            percentile=percentile,
                                            win_length=win_length,
                                            polyorder=polyorder)

        self._result_matrix = np.zeros((len(arr), len(arr)), dtype=np.float32)
        self._devices = None

        self.model = TELightning(len_time=len_time, dt=dt)
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
    def create_binned_array(self,
                            arr=None,
                            kp=None,
                            percentile=None,
                            win_length=None,
                            polyorder=None,
                            dtype=np.int32):

        kw = self.kernel_width(arr, kp, percentile)
        arr = savgol_filter(arr, win_length, polyorder)
        mins = np.min(arr, axis=1)
        arr = (arr.T - mins) // kw

        return arr.T.astype(dtype)

    def custom_collate(self, batch):
        n_devices = None

        if type(self._devices)==int:
            n_devices = self._devices
        elif type(self._devices)==list:
            n_devices = len(self._devices)

        pairs = [item[1] for item in batch]
        arr = batch[0][0]

        return arr, np.stack(pairs)

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
                            strategy="auto")

        trainer.predict(self.model, dloader_pair)

        if trainer.is_global_zero:
            results = self.model.return_result()
            return results


