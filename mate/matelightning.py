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
from mate.preprocess import DiscretizerFactory, SmootherFactory

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
                 binning_method = 'default',
                 binning_opt: dict = None,
                 smoothing_opt: dict = None,
                 len_time=None,
                 dt=1,
                 surrogate=False,
                 num_surrogate=1000,
                 threshold=0.05,
                 seed=1
                 ):
        # super().__init__()

        np.random.seed(seed)

        discretizer = DiscretizerFactory.create(binning_method=binning_method, binning_family=binning_opt, kp=kp)
        smoother = SmootherFactory.create(smoothing_opt=smoothing_opt)

        if smoother is None:
            print(f"[DISCRETIZER: {binning_method}, SMOOTHER: None]")
        else:
            print(f"[DISCRETIZER: {binning_method}, SMOOTHER: {smoothing_opt['method']}]")

        if smoother:
            arr = smoother.smoothing(arr)
        if discretizer:
            arr, n_bins = discretizer.binning(arr)

        self._devices = None

        self.model = TELightning(arr=arr,
                                 len_time=len_time,
                                 dt=dt,
                                 n_bins=n_bins,
                                 surrogate=surrogate,
                                 num_surrogate=num_surrogate,
                                 threshold=threshold,
                                 )
        self.dset_pair = PairDataSet(arr=arr, pairs=pairs)

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
            backend=None,
            devices=None,
            batch_size=None,
            num_workers=0):

        self._devices = devices

        dloader_pair = DataLoader(self.dset_pair,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=self.custom_collate)

        trainer = L.Trainer(accelerator=backend,
                            devices=devices,
                            num_nodes=1,
                            strategy="auto")


        trainer.predict(self.model, dloader_pair)

        if trainer.is_global_zero:
            results = self.model.return_result()

            return results


