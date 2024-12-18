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
from mate.utils import get_device_list, istarmap
from mate.preprocess import DiscretizerFactory, SmootherFactory

class MATE(object):
    def __init__(self,
                 backend=None,
                 device_ids=None,
                 procs_per_device=None,
                 batch_size=None,
                 kp=0.5,
                 num_kernels=1,
                 binning_method = 'default',
                 binning_opt: dict = None,
                 smoothing_opt: dict = None,
                 dt=1
                 ):

        self._batch_size = batch_size

        self._device = backend
        self._device_ids = device_ids
        self._procs_per_device = procs_per_device

        self._bin_arr = None
        self._result_matrix = None

        self._dt = dt

        self._discretizer = DiscretizerFactory.create(binning_method=binning_method, binning_family=binning_opt, kp=kp)
        self._smoother = SmootherFactory.create(smoothing_opt=smoothing_opt)

        if self._smoother is None:
            print(f"[DISCRETIZER: {binning_method}, SMOOTHER: None]")
        else:
            print(f"[DISCRETIZER: {binning_method}, SMOOTHER: {smoothing_opt['method']}]")

    def run(self,
            backend=None,
            device_ids=None,
            procs_per_device=None,
            batch_size=0,
            arr=None,
            pairs=None,
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

        if 'cpu' in backend or 'tenet' in backend:
            if procs_per_device > 1:
                raise Warning("CPU devices can only use one process per device")
            procs_per_device = 1

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

        n_bins = None
        if backend.lower() != "tenet":
            if self._smoother:
                arr = self._smoother.smoothing(arr)

            if self._discretizer:
                arr, n_bins = self._discretizer.binning(arr)

        self._result_matrix = np.zeros((len(arr), len(arr)), dtype=np.float32)

        n_pairs = len(pairs)

        n_process = len(device_ids)
        n_subpairs = math.ceil(n_pairs / n_process)
        n_procpairs = math.ceil(n_subpairs / procs_per_device)

        sub_batch = math.ceil(batch_size / procs_per_device)

        multiprocessing.set_start_method('spawn', force=True)

        processes = []
        t_beg_batch = time.time()

        if "cpu" in backend:
            print("[CPU device selected]")
            print("[Num. Processes: {}, Num. Pairs: {}, Num. Sub_Pair: {}, Batch Size: {}]".format(n_process, n_pairs,
                                                                                                   n_subpairs,
                                                                                                   batch_size))
        elif "tenet" in backend.lower():
            print("[TENET selected]")
            print("[Num. Processes: {}, Num. Pairs: {}, Num. Sub Pairs: {}]".format(n_process, n_pairs, n_subpairs))
        else:
            print("[GPU device selected]")
            print("[Num. GPUS: {}, Num. Pairs: {}, Num. GPU_Pairs: {}, Batch Size: {}, Process per device: {}]".format(
                n_process, n_pairs,
                n_subpairs, batch_size, procs_per_device))

        list_device = []
        list_subatch = []
        list_pairs = []
        list_arr = []
        list_dt = []
        list_surrogate = []
        list_numsurro = []
        list_threshold = []
        list_id = []

        if surrogate is True:
            # seeding for surrogate test before applying multiprocessing
            np.random.seed(seed)
            print("[Surrogate test option was activated]")
            print("[Number of surrogates] ", num_surrogate)
            print("[Threshold] ", threshold)

        cnt = 0
        for i, i_beg in enumerate(range(0, n_pairs, n_subpairs)):
            i_end = i_beg + n_subpairs

            for j, j_beg in enumerate(range(0, n_subpairs, n_procpairs)):
                t_beg = i_beg + j_beg
                t_end = t_beg + n_procpairs

                device_name = backend + ":" + str(device_ids[i])
                list_device.append(device_name)
                list_subatch.append(sub_batch)
                list_pairs.append(pairs[t_beg:t_end])
                list_arr.append(arr)
                list_dt.append(dt)
                list_surrogate.append(surrogate)
                list_numsurro.append(num_surrogate)
                list_threshold.append(threshold)
                list_id.append(cnt)
                cnt += 1

        pool = multiprocessing.Pool(processes=n_process * procs_per_device)

        if "tenet" in backend.lower():
            te = MATETENET()
            inputs = zip(list_pairs, list_arr, list_dt)
        else:
            te = TransferEntropy()
            inputs = zip(list_device,
                         list_subatch,
                         list_pairs,
                         list_arr,
                         list_dt,
                         list_surrogate,
                         list_numsurro,
                         list_threshold,
                         list_id)

        results = pool.starmap(te.solve, inputs)

        pool.close()
        pool.join()

        for result in results:
            pairs, entropies = result
            self._result_matrix[pairs[:, 0], pairs[:, 1]] = entropies

        print("Total processing elapsed time {}sec.".format(time.time() - t_beg_batch))

        return self._result_matrix
