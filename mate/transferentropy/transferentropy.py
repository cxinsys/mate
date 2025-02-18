import os
import os.path as osp
import time
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np
from tqdm import tqdm
from scipy.stats import norm

from mate.array import get_array_module

class TransferEntropy(object):
    @property
    def am(self):
        return self._am

    def compute_te(self,
                   bin_arrs=None,
                   t_pairs=None,
                   s_pairs=None,
                   tile_inds_pair=None,
                   dt=1,
                   len_time=None):

        target_arr = self.am.take(bin_arrs, t_pairs, axis=0)
        source_arr = self.am.take(bin_arrs, s_pairs, axis=0)
        # vals = self.am.stack((target_arr[:, (dt+1):, :],
        #                       target_arr[:, dt:-1, :],
        #                       source_arr[:, :-(dt+1), :]),
        #                      axis=3)
        vals = self.am.stack((target_arr[:, dt:, :],
                              target_arr[:, :-dt, :],
                              source_arr[:, :-dt, :]),
                             axis=3)

        t_vals = self.am.transpose(vals, axes=(2, 0, 1, 3))

        pair_vals = self.am.concatenate((tile_inds_pair[:, None], self.am.reshape(t_vals, (-1, 3))), axis=1)

        uvals_xt1_xt_yt, cnts_xt1_xt_yt = self.am.unique(pair_vals, return_counts=True, axis=0)

        uvals_xt1_xt, cnts_xt1_xt = self.am.unique(pair_vals[:, :-1], return_counts=True, axis=0)
        uvals_xt_yt, cnts_xt_yt = self.am.unique(self.am.take(pair_vals, self.am.array([0, 2, 3]), axis=1),
                                                 return_counts=True, axis=0)
        uvals_xt, cnts_xt = self.am.unique(self.am.take(pair_vals, self.am.array([0, 2]), axis=1), return_counts=True,
                                           axis=0)

        subuvals_xt1_xt, n_subuvals_xt1_xt = self.am.unique(uvals_xt1_xt_yt[:, :-1], return_counts=True, axis=0)
        subuvals_xt_yt, n_subuvals_xt_yt = self.am.unique(
            self.am.take(uvals_xt1_xt_yt, self.am.array([0, 2, 3]), axis=1), return_counts=True, axis=0)
        subuvals_xt, n_subuvals_xt = self.am.unique(self.am.take(uvals_xt1_xt_yt, self.am.array([0, 2]), axis=1),
                                                    return_counts=True, axis=0)

        cnts_xt1_xt = self.am.repeat(cnts_xt1_xt, n_subuvals_xt1_xt)

        cnts_xt_yt = self.am.repeat(cnts_xt_yt, n_subuvals_xt_yt)

        ind_xt_yt = self.am.lexsort(self.am.transpose(self.am.take(uvals_xt1_xt_yt, self.am.array([3, 2, 0]), axis=1), axes=None))
        ind2ori_xt_yt = self.am.argsort(ind_xt_yt)
        cnts_xt_yt = self.am.take(cnts_xt_yt, ind2ori_xt_yt)

        cnts_xt = self.am.repeat(cnts_xt, n_subuvals_xt)

        ind_xt = self.am.lexsort(self.am.transpose(self.am.take(uvals_xt1_xt_yt, self.am.array([2, 0]), axis=1), axes=None))
        ind2ori_xt = self.am.argsort(ind_xt)
        cnts_xt = self.am.take(cnts_xt, ind2ori_xt)

        # TE
        p_xt1_xt_yt = self.am.divide(cnts_xt1_xt_yt, (len_time - 1) * bin_arrs.shape[-1])
        # p_xt1_xt_yt = self.am.divide(cnts_xt1_xt_yt, (len_time - 1))
        numer = self.am.multiply(cnts_xt1_xt_yt, cnts_xt)
        denom = self.am.multiply(cnts_xt1_xt, cnts_xt_yt)
        fraction = self.am.divide(numer, denom)
        log_val = self.am.log2(fraction)
        entropies = self.am.multiply(p_xt1_xt_yt, log_val)

        uvals_tot, n_subuvals_tot = self.am.unique(uvals_xt1_xt_yt[:, 0], return_counts=True)
        final_bins = self.am.repeat(uvals_tot, n_subuvals_tot)
        final_bins = self.am.astype(x=final_bins, dtype='int32')
        entropy_final = self.am.bincount(final_bins, weights=entropies)

        return entropy_final

    def solve(self,
              backend='cpu',
              batch_size=None,
              pairs=None,
              bin_arrs=None,
              dt=1,
              surrogate=False,
              num_surrogate=1000,
              threshold=0.05,
              id=None,
              n_pairs=None,
              len_time=None,
              ):

        self._am = get_array_module(backend)

        if not batch_size:
            raise ValueError("batch size should be defined")

        if pairs is None:
            raise ValueError("pairs should be defined")

        if not n_pairs:
            n_pairs = len(pairs)


        if bin_arrs is None:
            raise ValueError("binned arrays should be defined")


        if not len_time:
            len_time = bin_arrs.shape[1]

        if not dt:
            dt = 1

        bin_arrs = self.am.array(bin_arrs, dtype=str(bin_arrs.dtype))
        g_pairs = self.am.array(pairs, dtype=str(pairs.dtype))

        entropy_final = []

        # print("[%s ID: %d, Batch #%d]" % (str(self.am.backend).upper(), self.am.device_id))
        for i_iter, i_beg in enumerate(tqdm(range(0, n_pairs, batch_size), desc=f"Process {id}", position=id, leave=True)):
            t_beg_batch = time.time()

            stime_preproc = time.time()

            i_end = i_beg + batch_size
            inds_pair = self.am.arange(len(g_pairs[i_beg:i_end]))

            t_pairs = g_pairs[i_beg:i_end, 0]
            s_pairs = g_pairs[i_beg:i_end, 1]

            tile_inds_pair = self.am.repeat(inds_pair, (len_time - 1)) # (pairs, time * kernel)
            tile_inds_pair = self.am.tile(tile_inds_pair, bin_arrs.shape[-1])

            entropies = self.compute_te(bin_arrs=bin_arrs,
                                        t_pairs=t_pairs,
                                        s_pairs=s_pairs,
                                        tile_inds_pair=tile_inds_pair,
                                        dt=dt,
                                        len_time=len_time)
            # end TE

            # if surrogate is True:
            #     surrogate_tes = []
            #     for i in tqdm(range(num_surrogate)):
            #         idx = np.random.rand(*bin_arrs.shape).argsort(axis=1)
            #         # shuffle array along trajectory axis
            #         bin_arrs = self.am.take_along_axis(bin_arrs, self.am.array(idx), axis=1)
            #
            #         entropy_surrogate = self.compute_te(bin_arrs=bin_arrs,
            #                                             t_pairs=t_pairs,
            #                                             s_pairs=s_pairs,
            #                                             tile_inds_pair=tile_inds_pair,
            #                                             n_bins=n_bins,
            #                                             dt=dt,
            #                                             len_time=len_time)
            #
            #         entropy_surrogate = self.am.asnumpy(entropy_surrogate)
            #
            #         surrogate_tes.append(entropy_surrogate)
            #
            #     print("[surrogate created]")
            #     surrogate_tes = np.array(surrogate_tes)
            #
            #     # # 2안: surrogate의 각 TE로부터 분포 구성 -> 상위 k% 값 추출
            #     means = np.mean(surrogate_tes, axis=0)
            #     std = np.std(surrogate_tes, axis=0)
            #     top_values = norm.ppf((1 - threshold), loc=means, scale=std)
            #
            #     # original te 값과 비교 -> original te < surrogate top te 이면 0
            #     entropies[self.am.asnumpy(entropies) <= top_values] = 0.0
            #     print("Number of entropies after eleminating FD")
            #     print(f'[Before] {len(entropies)}, [Num. zero val] {len(entropies) - len(np.nonzero(self.am.asnumpy(entropies))[0])}')

            entropy_final.extend(list(self.am.asnumpy(entropies)))

            # print("[%s ID: %d, Batch #%d] Batch processing elapsed time: %f" % (str(self.am.backend).upper(), self.am.device_id, i_iter + 1, time.time() - t_beg_batch))

        return pairs, np.array(entropy_final)