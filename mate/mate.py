import time
import math
from itertools import permutations
import multiprocessing
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np
# from KDEpy import TreeKDE, FFTKDE
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from mate.transferentropy import TransferEntropy, MATETENET
from mate.utils import get_device_list
from mate.preprocess import DiscretizerFactory

class MATE(object):
    def __init__(self,
                 backend=None,
                 device_ids=None,
                 procs_per_device=None,
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

        self._device = backend
        self._device_ids = device_ids
        self._procs_per_device = procs_per_device

        self._bin_arr = None
        self._result_matrix = None

        self._dt = dt

        self._discretizer = DiscretizerFactory.create(method=method, kp=kp)

    def run(self,
            backend=None,
            device_ids=None,
            procs_per_device=None,
            batch_size=0,
            arr=None,
            pairs=None,
            smooth_func=None,
            smooth_param=None,
            kw_smooth=True,
            data_smooth=False,
            dt=1,
            surrogate=False,
            num_surrogate=10,
            threshold=0.05,
            seed=1
            ):

        if not backend:
            if not self._device:
                self._device = backend = "cpu"
            backend = self._device

        if not device_ids:
            if not self._device_ids:
                if 'cpu' in backend:
                    self._device_ids = [0]
                    device_ids = [0]
                else:
                    self._device_ids = get_device_list()
            device_ids = self._device_ids

        if not procs_per_device:
            if not self._procs_per_device:
                self._procs_per_device = 1
            procs_per_device = self._procs_per_device

        if 'cpu' in backend:
            if procs_per_device > 1:
                raise ValueError("CPU devices can only use one process per device")

        if type(device_ids) is int:
            list_device_ids = [x for x in range(device_ids)]
            device_ids = list_device_ids

        if not batch_size and backend.lower() != "tenet":
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

        if backend.lower() != "tenet":
            arr, n_bins = self._discretizer.binning(arr)

        self._result_matrix = np.zeros((len(arr), len(arr)), dtype=np.float32)

        n_pairs = len(pairs)

        n_process = len(device_ids)
        n_subpairs = math.ceil(n_pairs / n_process)
        n_procpairs = math.ceil(n_subpairs / procs_per_device)

        sub_batch = math.ceil(batch_size / procs_per_device)

        multiprocessing.set_start_method('spawn')

        processes = []
        t_beg_batch = time.time()
        if "cpu" in backend:
            print("[CPU device selected]")
            print("[Num. Processes: {}, Num. Pairs: {}, Num. Sub_Pair: {}, Batch Size: {}]".format(n_process, n_pairs,
                                                                                                 n_subpairs, batch_size))
        elif "tenet" in backend.lower():
            print("[TENET selected]")
            print("[Num. Processes: {}, Num. Pairs: {}, Num. Sub Pairs: {}]".format(n_process, n_pairs, n_subpairs))
        else:
            print("[GPU device selected]")
            print("[Num. GPUS: {}, Num. Pairs: {}, Num. GPU_Pairs: {}, Batch Size: {}, Process per device: {}]".format(n_process, n_pairs,
                                                                                               n_subpairs, batch_size, procs_per_device))
        list_device = []
        list_subatch = [sub_batch for i in range(n_process)]
        list_pairs = []
        list_arr = [arr for i in range(n_process)]
        list_dt = [dt for i in range(n_process)]
        list_surrogate = [surrogate for i in range(n_process)]
        list_numsurro = [num_surrogate for i in range(n_process)]
        list_threshold = [threshold for i in range(n_process)]

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

                _process = None

                device_name = backend + ":" + str(device_ids[i])
                list_device.append(device_name)
                list_pairs.append(pairs[t_beg:t_end])
                # processes.append(_process) # test

        pool = multiprocessing.Pool(processes=n_process)

        if "tenet" in backend.lower():
            te = MATETENET()
            inputs = zip(list_pairs, list_arr, list_dt)
        else:
            list_nbins = [n_bins for i in range(n_process)]

            te = TransferEntropy()
            inputs = zip(list_device,
                         list_subatch,
                         list_pairs,
                         list_arr,
                         list_nbins,
                         list_dt,
                         list_surrogate,
                         list_numsurro,
                         list_threshold)

        results = pool.starmap(te.solve, inputs)

        pool.close()
        pool.join()

        for result in results:
            pairs, entropies = result
            self._result_matrix[pairs[:, 0], pairs[:, 1]] = entropies

        print("Total processing elapsed time {}sec.".format(time.time() - t_beg_batch))

        return self._result_matrix

