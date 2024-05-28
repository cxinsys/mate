import os
import os.path as osp
import time
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np

from mate.array import get_array_module

class TransferEntropy(object):
    def __init__(self,
                 device=None,
                 batch_size=None,
                 pairs=None,
                 n_pairs=None,
                 inds_pair=None,
                 bin_arrs=None,
                 n_bins=None,
                 len_time=None,
                 shm_name=None,
                 result_matrix=None,
                 sem=None,
                 dt=1
                 ):

        self._am = get_array_module(device)

        self._batch_size = batch_size
        self._pairs = pairs
        self._n_pairs = n_pairs

        self._inds_pair = inds_pair
        self._bin_arrs = bin_arrs
        self._n_bins = n_bins

        if inds_pair is not None:
            with self.am:
                self._inds_pair = self.am.array(inds_pair, dtype=inds_pair.dtype)

        if bin_arrs is not None:
            with self.am:
                self._bin_arrs = bin_arrs # list of binned arrays

        self._len_time = len_time

        self._shm_name = shm_name
        self._result_matrix = result_matrix
        self._sem = sem

        self._dt = dt

        # self.fpath_log = osp.join('./', 'elapsed_time_per_task.csv')
        #
        # if not osp.exists(self.fpath_log):
        #     with open(self.fpath_log, 'w') as f:
        #         f.write("Task, " \
        #                 "Elapsed time \n")

    @property
    def am(self):
        return self._am

    def solve(self,
              batch_size=None,
              pairs=None,
              bin_arrs=None,
              n_bins=None,
              shm_name=None,
              result_matrix=None,
              sem=None,
              dt=1,
              n_pairs=None,
              inds_pair=None,
              len_time=None,
              ):

        if not batch_size:
            if not self._batch_size:
                raise ValueError("batch size should be defined")
            batch_size = self._batch_size

        if pairs is None:
            if self._pairs is None:
                raise ValueError("pairs should be defined")
            pairs = self._pairs

        if not n_pairs:
            if not self._n_pairs:
                self._n_pairs = n_pairs = len(pairs)
            n_pairs = self._n_pairs

        if inds_pair is None:
            if self._inds_pair is None:
                self._inds_pair = inds_pair = np.arange(batch_size)
            inds_pair = self._inds_pair

        if bin_arrs is None:
            if self._bin_arrs is None:
                raise ValueError("binned arrays should be defined")
            bin_arrs = self._bin_arrs

        if n_bins is None:
            if self._n_bins is None:
                raise ValueError("pairs should be defined")
            n_bins = self._n_bins

        if not len_time:
            if not self._len_time:
                self._len_time = len_time = bin_arrs.shape[1]
                # self._len_time = len_time = bin_arrs[0].shape[1]
            len_time = self._len_time

        if not shm_name:
            if not self._shm_name:
                raise ValueError("shared memory name should be defined")
            shm_name = self._shm_name

        if result_matrix is None:
            if self._result_matrix is None:
                raise ValueError("result matrix should be defined")
            result_matrix = self._result_matrix

        if not sem:
            if not self._sem:
                raise ValueError("semaphore should be defined")
            sem = self._sem

        if not dt:
            if not self._dt:
                self._dt = dt = 1
            dt = self._dt

        bin_arrs = self.am.array(bin_arrs, dtype=bin_arrs.dtype)
        g_pairs = self.am.array(pairs, dtype=pairs.dtype)

        for i_iter, i_beg in enumerate(range(0, n_pairs, batch_size)):
            t_beg_batch = time.time()

            print("[%s ID: %d, Batch #%d]" % (str(self.am.device).upper(), self.am.device_id, i_iter + 1))

            stime_preproc = time.time()

            i_end = i_beg + batch_size
            inds_pair = self.am.arange(len(g_pairs[i_beg:i_end]))

            t_pairs = g_pairs[i_beg:i_end, 0]
            s_pairs = g_pairs[i_beg:i_end, 1]

            tile_inds_pair = self.am.repeat(inds_pair, (len_time - 1)) # (pairs, time * kernel)
            tile_inds_pair = self.am.tile(tile_inds_pair, bin_arrs.shape[-1])

            target_arr = self.am.take(bin_arrs, t_pairs, axis=0)
            source_arr = self.am.take(bin_arrs, s_pairs, axis=0)
            vals = self.am.stack((target_arr[:, dt:, :],
                                  target_arr[:, :-dt, :],
                                  source_arr[:, :-dt, :]),
                                  axis=3)


            t_vals = self.am.transpose(vals, axes=(2, 0, 1, 3))

            pair_vals = self.am.concatenate((tile_inds_pair[:, None], self.am.reshape(t_vals, (-1, 3))), axis=1)

            # stime_imaginary = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("Preprocess, " \
            #             f"{stime_imaginary - stime_preproc} \n")

            # 허수 제거
            n_bins = self.am.array(n_bins, dtype=n_bins.dtype)
            n_bins = self.am.take(n_bins, t_pairs, axis=0)
            n_bins = self.am.repeat(n_bins, (len_time - 1))
            n_bins = self.am.tile(n_bins, bin_arrs.shape[-1])

            # left_bools = self.am.array(
            #     self.am.logical_and(
            #         self.am.greater_equal(pair_vals[:, 2], 0),
            #         self.am.less(pair_vals[:, 2], n_bins)
            #     )
            # )
            left_bools = self.am.logical_and(
                self.am.greater_equal(pair_vals[:, 2], 0),
                self.am.less(pair_vals[:, 2], n_bins)
            )

            left_inds = self.am.where(left_bools)[0]

            pair_vals = self.am.take(pair_vals, left_inds, axis=0)
            # 허수 제거

            # stime_unique = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("Remove imaginary, " \
            #             f"{stime_unique - stime_imaginary} \n")

            uvals_xt1_xt_yt, cnts_xt1_xt_yt = self.am.unique(pair_vals, return_counts=True, axis=0)

            uvals_xt1_xt, cnts_xt1_xt = self.am.unique(pair_vals[:, :-1], return_counts=True, axis=0)
            uvals_xt_yt, cnts_xt_yt = self.am.unique(self.am.take(pair_vals, self.am.array([0, 2, 3]), axis=1),
                                                     return_counts=True, axis=0)
            uvals_xt, cnts_xt = self.am.unique(self.am.take(pair_vals, self.am.array([0, 2]), axis=1), return_counts=True,
                                               axis=0)

            subuvals_xt1_xt, n_subuvals_xt1_xt = self.am.unique(uvals_xt1_xt_yt[:, :-1], return_counts=True, axis=0)
            subuvals_xt_yt, n_subuvals_xt_yt = self.am.unique(self.am.take(uvals_xt1_xt_yt, self.am.array([0, 2, 3]), axis=1), return_counts=True, axis=0)
            subuvals_xt, n_subuvals_xt = self.am.unique(self.am.take(uvals_xt1_xt_yt, self.am.array([0, 2]), axis=1), return_counts=True, axis=0)

            # s_time = time.time()
            # tmp_cnts_xt1_xt = self.am.concatenate([self.am.broadcast_to(cnt, self.am.take(n_subuvals_xt1_xt, i).item()) for i, cnt in enumerate(cnts_xt1_xt)])
            # print(time.time() - s_time)

            # stime_repeat1 = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("Uniques, " \
            #             f"{stime_repeat1 - stime_unique} \n")
            cnts_xt1_xt = self.am.repeat(cnts_xt1_xt, n_subuvals_xt1_xt)

            # stime_repeat2 = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("Repeat1, " \
            #             f"{stime_repeat2 - stime_repeat1} \n")
            cnts_xt_yt = self.am.repeat(cnts_xt_yt, n_subuvals_xt_yt)

            # stime_sort1 = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("Repeat2, " \
            #             f"{stime_sort1 - stime_repeat2} \n")
            ind_xt_yt = self.am.lexsort(self.am.take(uvals_xt1_xt_yt, self.am.array([3, 2, 0]), axis=1).T)
            ind2ori_xt_yt = self.am.argsort(ind_xt_yt)
            cnts_xt_yt = self.am.take(cnts_xt_yt, ind2ori_xt_yt)

            # stime_repeat3 = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("Sort1, " \
            #             f"{stime_repeat3 - stime_sort1} \n")
            cnts_xt = self.am.repeat(cnts_xt, n_subuvals_xt)

            # stime_sort2 = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("Repeat3, " \
            #             f"{stime_sort2 - stime_repeat3} \n")
            ind_xt = self.am.lexsort(self.am.take(uvals_xt1_xt_yt, self.am.array([2, 0]), axis=1).T)
            ind2ori_xt = self.am.argsort(ind_xt)
            cnts_xt = self.am.take(cnts_xt, ind2ori_xt)

            # stime_te = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("Sort2, " \
            #             f"{stime_te - stime_sort2} \n")
            # TE
            p_xt1_xt_yt = self.am.divide(cnts_xt1_xt_yt, (len_time - 1) * bin_arrs.shape[-1])
            # p_xt1_xt_yt = self.am.divide(cnts_xt1_xt_yt, (len_time - 1))

            numer = self.am.multiply(cnts_xt1_xt_yt, cnts_xt)
            denom = self.am.multiply(cnts_xt1_xt, cnts_xt_yt)
            fraction = self.am.divide(numer, denom)
            log_val = self.am.log2(fraction)
            entropies = self.am.multiply(p_xt1_xt_yt, log_val)

            # stime_bincount = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("TE, " \
            #             f"{stime_bincount - stime_te} \n")
            uvals_tot, n_subuvals_tot = self.am.unique(uvals_xt1_xt_yt[:, 0], return_counts=True)
            final_bins = self.am.repeat(uvals_tot, n_subuvals_tot)
            final_bins = self.am.astype(x=final_bins, dtype='int32')
            entropy_final = self.am.bincount(final_bins, weights=entropies)

            # etime = time.time()
            # with open(self.fpath_log, 'a') as f:
            #     f.write("Bincount, " \
            #             f"{etime - stime_bincount} \n")

            # # LocalTE
            # numer = self.am.multiply(cnts_xt1_xt_yt, cnts_xt)
            # denom = self.am.multiply(cnts_xt1_xt, cnts_xt_yt)
            # fraction = self.am.divide(numer, denom)
            # log_val = self.am.log2(fraction)
            #
            # uvals_tot, n_subuvals_tot = self.am.unique(uvals_xt1_xt_yt[:, 0], return_counts=True)
            # final_bins = self.am.repeat(uvals_tot, n_subuvals_tot)
            # final_bins = self.am.astype(x=final_bins, dtype='int32')
            # entropies = self.am.bincount(final_bins, weights=log_val)
            #
            # entropy_final = self.am.divide(entropies, len_time - 1)

            # end TE
            entropy_final = self.am.asnumpy(entropy_final)

            sem.acquire()

            new_shm = shared_memory.SharedMemory(name=shm_name)
            tmp_arr = np.ndarray(result_matrix.shape, dtype=result_matrix.dtype, buffer=new_shm.buf)
            tmp_arr[pairs[i_beg:i_end, 0], pairs[i_beg:i_end, 1]] = entropy_final

            new_shm.close()

            sem.release()

            print("[%s ID: %d, Batch #%d] Batch processing elapsed time: %f" % (str(self.am.device).upper(), self.am.device_id, i_iter + 1, time.time() - t_beg_batch))