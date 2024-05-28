import time
import math
from itertools import permutations
import multiprocessing
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np
# from KDEpy import TreeKDE, FFTKDE
from tqdm import tqdm

from mate.transferentropy import TransferEntropy
from mate.utils import get_device_list

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
                 method='pushing',
                 percentile=0,
                 smooth_func=None,
                 smooth_param=None,
                 dt=1
                 ):

        self._kp = kp
        self._num_kernels = num_kernels
        self._method = method
        self._percentile = percentile
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
    def create_kde_array(self,
                         kp=None,
                         num_kernels=None,
                         method='interpolation',
                         dtype=np.int32
                         ):

        if not kp:
            kp = self._kp
        if not num_kernels:
            num_kernels = self._num_kernels

        arr = self._arr

        stds = np.std(arr, axis=1, ddof=1)
        mins = np.min(arr, axis=1)
        maxs = np.max(arr, axis=1)

        n_bins = np.ceil((maxs - mins) / stds).T.astype(dtype)

        arrs = []

        print(f"[Selected Method: {method.upper()}]")

        if method=='interpolation':
            bin_arr = ((arr.T - mins) / stds).T
            mid_arr = (bin_arr[:, :-1] + bin_arr[:, 1:]) / 2

            inter_arr = np.zeros((len(bin_arr), len(bin_arr[0])+len(mid_arr[0])))

            inter_arr[:, ::2] = bin_arr
            inter_arr[:, 1::2] = mid_arr

            # Int Bin
            # inter_arr = np.floor(inter_arr).astype(dtype)

            # Float Bin
            inter_arr = inter_arr.astype(np.float32)

            inter_arr = np.where(inter_arr < 0, 0, inter_arr)
            inter_arr = np.where(inter_arr >= n_bins, n_bins - 1, inter_arr)

            print(f"Interpolation applied. Increased data length from {len(arr[0])} to {len(inter_arr[0])}.")

            arrs = inter_arr[..., None]

        elif method=='tagging':
            print(f"Number of binned arrays for increasing pattern: {num_kernels}")

            for i in range(num_kernels):
                if i % 2 == 1: # odd
                    bin_arr = np.floor((arr.T - (mins + ((i//2 + i%2) * kp * stds))) / stds).T.astype(dtype)
                else:
                    bin_arr = np.floor((arr.T - (mins - (i//2 * kp * stds))) / stds).T.astype(dtype)


                bin_arr = np.where(bin_arr<0, 0, bin_arr)
                bin_arr = np.where(bin_arr>=n_bins, n_bins-1, bin_arr)

                bin_maxs = np.max(bin_arr, axis=1)

                coeff = (i + 1) * 10 ** np.ceil(np.log10(bin_maxs))

                bin_arr += coeff[..., None].astype(dtype)

                arrs.append(bin_arr)

            arrs = np.stack(arrs, axis=2)

        elif method=='shifting':
            print(f"[Num. Kernel: {num_kernels}, Kernel Width: {kp}]")

            for i in range(num_kernels):
                if i % 2 == 1:  # odd
                    bin_arr = np.floor((arr.T - (mins + ((i // 2 + i % 2) * kp * stds))) / stds).T.astype(dtype) # pull
                else:
                    bin_arr = np.floor((arr.T - (mins - (i // 2 * kp * stds))) / stds).T.astype(dtype) # push

                bin_arr = bin_arr.astype(dtype)

                arrs.append(bin_arr)
            arrs = np.stack(arrs, axis=2)

        elif method == 'pushing':
            print(f"[Kernel Width: {kp}]")

            bin_arr = np.floor((arr.T - (mins - (kp * stds))) / stds).T.astype(dtype)

            arrs = bin_arr[..., None]

        elif method == 'pulling':
            print(f"[Kernel Width: {kp}]")

            bin_arr = np.floor((arr.T - (mins + (kp * stds))) / stds).T.astype(dtype)

            arrs = bin_arr[..., None]

        elif method == 'pushpull':
            print(f"[Kernel Width: {kp}]")

            bin_arr = np.floor((arr.T - (mins + (kp * stds))) / stds).T.astype(dtype)
            arrs.append(bin_arr)
            bin_arr = np.floor((arr.T - (mins - (kp * stds))) / stds).T.astype(dtype)
            arrs.append(bin_arr)

            arrs = np.stack(arrs, axis=2)

        else:
            print("Default Binning")
            bin_arr = np.floor((arr.T - mins) / stds).T.astype(dtype)
            arrs = bin_arr[..., None]

        return arrs, n_bins


    # multiprocessing worker(calculate te)

    def run(self,
            device=None,
            device_ids=None,
            procs_per_device=None,
            batch_size=None,
            arr=None,
            pairs=None,
            kp=None,
            num_kernels=None,
            method=None,
            percentile=None,
            smooth_func=None,
            smooth_param=None,
            kw_smooth=True,
            data_smooth=False,
            dt=1
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

        if not kp:
            kp = self._kp

        if not num_kernels:
            num_kernels = self._num_kernels

        if not method:
            method = self._method
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

        arr, n_bins = self.create_kde_array(kp=kp,
                                    num_kernels=num_kernels,
                                    method=method)
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
                                                          dt))
                processes.append(_process)
                _process.start()

        for _process in processes:
            _process.join()

        print("Total processing elapsed time {}sec.".format(time.time() - t_beg_batch))

        self._result_matrix = np_shm.copy()

        shm.close()
        shm.unlink()

        return self._result_matrix

