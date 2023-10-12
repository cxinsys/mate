import time
import math
from itertools import permutations
import multiprocessing
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np

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
                 smooth_func=None,
                 smooth_param=None
                 ):

        self._kp = kp
        self._percentile = percentile
        self._batch_size = batch_size

        self._smooth_func = smooth_func
        self._smooth_param = smooth_param

        self._device = device
        self._device_ids = device_ids

        self._arr = arr
        self._pairs = pairs

        self._bin_arr = None
        self._result_matrix = None

    # calculate kernel width

    def kernel_width(self, arr=None, kp=None, percentile=None):
        if arr is None:
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
                            smooth_func=None,
                            smooth_param=None,
                            kw_smooth=True,
                            data_smooth=True,
                            dtype=np.int32):

        if not kp:
            kp = self._kp

        if not percentile:
            percentile = self._percentile

        if not smooth_func:
            smooth_func = self._smooth_func

        if not smooth_param:
            smooth_param = self._smooth_param

        if self._bin_arr is None:
            arr = self._arr
            smooth_arr = None

            if smooth_func is not None:
                print(f"[Applying Smooth Function] Kernel smoothing: {kw_smooth}, Data smoothing: {data_smooth}")
                if type(smooth_param)==tuple:
                    smooth_arr = smooth_func(arr, *smooth_param)
                elif type(smooth_param)==dict:
                    smooth_arr = smooth_func(arr, **smooth_param)
                else:
                    raise ValueError("Function parameter type must be tuple or dictionary")
                # arr = savgol_filter(arr, win_length, polyorder)
            else:
                kw_smooth = False
                data_smooth = False

            if kw_smooth==True:
                kw = self.kernel_width(smooth_arr, kp, percentile)
            else:
                kw = self.kernel_width(arr, kp, percentile)

            if data_smooth==True:
                mins = np.min(smooth_arr, axis=1)
                self._bin_arr = smooth_arr.copy()
                self._bin_arr = (self._bin_arr.T - mins) // kw
                self._bin_arr = self._bin_arr.T.astype(dtype)
            else:
                mins = np.min(arr, axis=1)
                self._bin_arr = arr.copy()
                self._bin_arr = (self._bin_arr.T - mins) // kw
                self._bin_arr = self._bin_arr.T.astype(dtype)

            del(arr)
            del(smooth_arr)

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
            smooth_func=None,
            smooth_param=None,
            kw_smooth=True,
            data_smooth=False,
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

        if type(device_ids) is int:
            list_device_ids = [x for x in range(device_ids)]
            device_ids = list_device_ids

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
                self._pairs = np.asarray(tuple(self._pairs), dtype=np.int32)
            pairs = self._pairs

        if not kp:
            kp = self._kp

        if not percentile:
            percentile = self._percentile

        if not smooth_func:
            smooth_func = self._smooth_func

        if not smooth_param:
            smooth_param = self._smooth_param

        self._arr = arr
        self._pairs = pairs

        arr = self.create_binned_array(kp=kp,
                                       percentile=percentile,
                                       smooth_func=smooth_func,
                                       smooth_param=smooth_param,
                                       kw_smooth=kw_smooth,
                                       data_smooth=data_smooth,
                                       )

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

