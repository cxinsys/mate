import time
import math
from itertools import permutations
import multiprocessing
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np
from scipy.signal import savgol_filter

from mate.transferentropy import TransferEntropy
from mate.utils import get_gpu_list

class MATE(object):
    def __init__(self,
                 device=None,
                 device_ids=None,
                 arr=None,
                 pairs=None,
                 batch_size=None,
                 kp=0.5,
                 percentile=0,
                 win_length=10,
                 polyorder=3,
                 ):

        self._kp = kp
        self._percentile = percentile
        self._win_length = win_length
        self._polyorder = polyorder
        self._batch_size = batch_size

        self._device = device
        self._device_ids = device_ids

        self._arr = arr
        self._pairs = pairs

        self._bin_arr = None
        self._result_matrix = None

    # calculate kernel width

    def kernel_width(self, kp=None, percentile=None):
        arr = self._arr

        if percentile > 0:
            arr2 = arr.copy()
            arr2.sort(axis=1)

            i_beg = int(arr2.shape[1] / 100 * percentile)

            std = np.std(arr2[:, i_beg:-i_beg], axis=1, ddof=1)
        else:
            std = np.std(arr, axis=1, ddof=1)

        kw = kp * std
        kw[kw == 0] = 1
        return kw

    # binning

    def create_binned_array(self,
                            kp=None,
                            percentile=None,
                            win_length=None,
                            polyorder=None,
                            dtype=np.int32):

        if not kp:
            kp = self._kp

        if not percentile:
            percentile = self._percentile

        if not win_length:
            win_length = self._win_length

        if not polyorder:
            polyorder = self._polyorder

        if self._bin_arr is None:
            arr = self._arr

            kw = self.kernel_width(kp, percentile)

            arr = savgol_filter(arr, win_length, polyorder)

            mins = np.min(arr, axis=1)
            # maxs = np.max(arr, axis=1)

            self._bin_arr = arr.copy()
            self._bin_arr = (arr.T - mins) // kw
            self._bin_arr = self._bin_arr.T.astype(dtype)

        return self._bin_arr

    # multiprocessing worker(calculate tenet)

    def run(self,
            device=None,
            device_ids=None,
            batch_size=None,
            arr=None,
            pairs=None,
            kp=None,
            percentile=None,
            win_length=None,
            polyorder=None,
            ):

        if not device:
            if not self._device:
                self._device = device = "cpu"
            device = self._device

        if not device_ids:
            if not self._device_ids:
                if 'cpu' in device:
                    self._device_ids = [0]
                    device_ids = [0]
                else:
                    self._device_ids = get_gpu_list()
            device_ids = self._device_ids

        if not batch_size:
            if not self._batch_size:
                raise ValueError("batch size should be refined")
            batch_size = self._batch_size

        if arr is None:
            if self._arr is None:
                raise ValueError("data should be refined")
            arr = self._arr

        if pairs is None:
            if self._pairs is None:
                self._pairs = permutations(range(len(arr)), 2)
                self._pairs = np.asarray(tuple(pairs), dtype=np.int32)
            pairs = self._pairs

        if not kp:
            kp = self._kp

        if not percentile:
            percentile = self._percentile

        if not win_length:
            win_length = self._win_length

        if not polyorder:
            polyorder = self._polyorder

        self._arr = arr
        self._pairs = pairs

        arr = self.create_binned_array(kp=kp,
                                       percentile=percentile,
                                       win_length=win_length,
                                       polyorder=polyorder)

        n_pairs = len(pairs)

        tmp_rm = np.zeros((len(arr), len(arr)), dtype=np.float32)

        n_process = len(device_ids)
        n_subpairs = math.ceil(n_pairs / n_process)

        multiprocessing.set_start_method('spawn', force=True)
        shm = shared_memory.SharedMemory(create=True, size=tmp_rm.nbytes)
        np_shm = np.ndarray(tmp_rm.shape, dtype=tmp_rm.dtype, buffer=shm.buf)
        np_shm[:] = tmp_rm[:]

        sem = Semaphore()

        processes = []
        t_beg_batch = time.time()
        if "cpu" in device:
            print("[CPU device selected]")
            print("[Num. Process: {}, Num. Pairs: {}, Num. Sub_Pair: {}, Batch Size: {}]".format(n_process, n_pairs,
                                                                                                 n_subpairs, batch_size))
        else:
            print("[GPU device selected]")
            print("[Num. GPUS: {}, Num. Pairs: {}, Num. GPU_Pairs: {}, Batch Size: {}]".format(n_process, n_pairs,
                                                                                               n_subpairs, batch_size))

        for i, i_beg in enumerate(range(0, n_pairs, n_subpairs)):
            i_end = i_beg + n_subpairs

            device_name = device + ":" + str(device_ids[i])
            # print("tenet device: {}".format(device_name))
            te = TransferEntropy(device=device_name)

            _process = Process(target=te.solve, args=(batch_size,
                                                      pairs[i_beg:i_end],
                                                      arr,
                                                      shm.name,
                                                      np_shm,
                                                      sem))
            processes.append(_process)
            _process.start()

        for _process in processes:
            _process.join()

        print("Total processing elapsed time {}sec.".format(time.time() - t_beg_batch))

        self._result_matrix = np_shm.copy()

        shm.close()
        shm.unlink()

        return self._result_matrix

