import time
import math
from itertools import permutations
import multiprocessing
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np
# from KDEpy import TreeKDE, FFTKDE
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from mate.transferentropy import TransferEntropy
from mate.utils import get_device_list
from mate.preprocess import DiscretizerFactory

class MATE(object):
    def __init__(self,
                 device=None,
                 device_ids=None,
                 procs_per_device=None,
                 arr=None,
                 pairs=None,
                 batch_size=None,
                 kp=0.5,
                 num_kernels=1,
                 method='default',
                 percentile=0,
                 smooth_func=None,
                 smooth_param=None,
                 dt=1
                 ):

        # self._kp = kp
        # self._num_kernels = num_kernels
        # self._method = method
        # self._percentile = percentile
        self._batch_size = batch_size

        self._smooth_func = smooth_func
        self._smooth_param = smooth_param

        self._device = device
        self._device_ids = device_ids
        self._procs_per_device = procs_per_device

        self._arr = arr
        self._pairs = pairs

        self._bin_arr = None
        self._result_matrix = None

        self._dt = dt

        self._discretizer = DiscretizerFactory.create(method=method, kp=kp)

    def run(self,
            device=None,
            device_ids=None,
            procs_per_device=None,
            batch_size=None,
            arr=None,
            pairs=None,
            smooth_func=None,
            smooth_param=None,
            kw_smooth=True,
            data_smooth=False,
            dt=1,
            surrogate=False,
            num_surrogate=1000,
            threshold=0.05,
            seed=1
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
                    self._device_ids = get_device_list()
            device_ids = self._device_ids

        if not procs_per_device:
            if not self._procs_per_device:
                self._procs_per_device = 1
            procs_per_device = self._procs_per_device

        if 'cpu' in device:
            if procs_per_device > 1:
                raise ValueError("CPU devices can only use one process per device")

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

        if not dt:
            dt = self._dt

        # if not percentile:
        #     percentile = self._percentile
        # if not smooth_func:
        #     smooth_func = self._smooth_func
        #
        # if not smooth_param:
        #     smooth_param = self._smooth_param

        self._arr = arr
        self._pairs = pairs

        arr, n_bins = self._discretizer.binning(arr)
        tmp_rm = np.zeros((len(arr), len(arr)), dtype=np.float32)

        n_pairs = len(pairs)

        n_process = len(device_ids)
        n_subpairs = math.ceil(n_pairs / n_process)
        n_procpairs = math.ceil(n_subpairs / procs_per_device)

        sub_batch = math.ceil(batch_size / procs_per_device)

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
            print("[Num. GPUS: {}, Num. Pairs: {}, Num. GPU_Pairs: {}, Batch Size: {}, Process per device: {}]".format(n_process, n_pairs,
                                                                                               n_subpairs, batch_size, procs_per_device))

        if surrogate is True:
            # seeding for surrogate test before applying multiprocessing
            np.random.seed(seed)
            print("[Surrogate test option was activated]")
            print("[Number of surrogates] ", num_surrogate)
            print("[Threshold] ", threshold)

        for i, i_beg in enumerate(range(0, n_pairs, n_subpairs)):
            i_end = i_beg + n_subpairs

            for j, j_beg in enumerate(range(0, n_subpairs, n_procpairs)):
                t_beg = i_beg + j_beg
                t_end = t_beg + n_procpairs

                device_name = device + ":" + str(device_ids[i])
                # print("tenet device: {}".format(device_name))

                te = TransferEntropy(device=device_name)

                _process = Process(target=te.solve, args=(sub_batch,
                                                          pairs[t_beg:t_end],
                                                          arr,
                                                          n_bins,
                                                          shm.name,
                                                          np_shm,
                                                          sem,
                                                          dt,
                                                          surrogate,
                                                          num_surrogate,
                                                          threshold))
                processes.append(_process)
                _process.start()


        for _process in processes:
            _process.join()

        print("Total processing elapsed time {}sec.".format(time.time() - t_beg_batch))

        self._result_matrix = np_shm.copy()

        shm.close()
        shm.unlink()

        return self._result_matrix

