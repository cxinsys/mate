import time

import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl


def lexsort(keys, dim=-1):
    if keys.ndim < 2:
        raise ValueError(f"keys must be at least 2 dimensional, but {keys.ndim=}.")
    if len(keys) == 0:
        raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")

    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.index_select(dim, k.index_select(dim, idx).argsort(dim=dim, stable=True))

    return idx

class TELightning(pl.LightningModule):
    def __init__(self, len_time=None, dt=1):
        super().__init__()
        self._len_time = len_time
        self._dt = dt
        self._result_matrix = None
        self._batch_predict_ef = []
        self._batch_predict_pairs = []
    def forward(self, arr, pairs):
        self._result_matrix = np.zeros((len(arr), len(arr)), dtype=np.float32)

        if self._len_time is None:
            self._len_time = len(arr[1])

        arr = torch.tensor(arr, dtype=torch.float32)
        pairs = torch.tensor(pairs, dtype=torch.int32)

        # print("array shape: ", arr.shape)
        # print("pairs shape: ", pairs.shape)

        inds_pair = torch.arange(len(pairs))
        tile_inds_pair = torch.repeat_interleave(inds_pair, self._len_time-1)

        # arr = torch.tensor(arr, dtype=arr.dtype)
        target_arr = torch.index_select(arr, 0, pairs[:, 0])
        source_arr = torch.index_select(arr, 0, pairs[:, 1])

        vals = torch.stack((target_arr[:, self._dt:],
                           target_arr[:, :-self._dt],
                           source_arr[:, :-self._dt]),
                           dim=2)

        pair_vals = torch.concatenate((tile_inds_pair[:, None], torch.reshape(vals, (-1, 3))), dim=1)

        uvals_xt1_xt_yt, cnts_xt1_xt_yt = torch.unique(pair_vals, return_counts=True, dim=0)
        uvals_xt1_xt, cnts_xt1_xt = torch.unique(pair_vals[:, :-1], return_counts=True, dim=0)
        uvals_xt_yt, cnts_xt_yt = torch.unique(torch.index_select(pair_vals, 1, torch.tensor([0, 2, 3])),
                                                 return_counts=True, dim=0)
        uvals_xt, cnts_xt = torch.unique(torch.index_select(pair_vals, 1, torch.tensor([0, 2])),
                                               return_counts=True, dim=0)

        subuvals_xt1_xt, n_subuvals_xt1_xt = torch.unique(uvals_xt1_xt_yt[:, :-1], return_counts=True, dim=0)
        subuvals_xt_yt, n_subuvals_xt_yt = torch.unique(torch.index_select(uvals_xt1_xt_yt, 1, torch.tensor([0, 2, 3])),
                                                          return_counts=True, dim=0)
        subuvals_xt, n_subuvals_xt = torch.unique(torch.index_select(uvals_xt1_xt_yt, 1, torch.tensor([0, 2])),
                                                    return_counts=True, dim=0)

        cnts_xt1_xt = torch.repeat_interleave(cnts_xt1_xt, n_subuvals_xt1_xt)

        cnts_xt_yt = torch.repeat_interleave(cnts_xt_yt, n_subuvals_xt_yt)
        ind_xt_yt = lexsort(torch.index_select(uvals_xt1_xt_yt, 1, torch.tensor([3, 2, 0])).T)
        ind2ori_xt_yt = torch.argsort(ind_xt_yt)
        cnts_xt_yt = torch.take(cnts_xt_yt, ind2ori_xt_yt)

        cnts_xt = torch.repeat_interleave(cnts_xt, n_subuvals_xt)
        ind_xt = lexsort(torch.index_select(uvals_xt1_xt_yt, 1, torch.tensor([2, 0])).T)
        ind2ori_xt = torch.argsort(ind_xt)
        cnts_xt = torch.take(cnts_xt, ind2ori_xt)

        p_xt1_xt_yt = torch.divide(cnts_xt1_xt_yt, (self._len_time - 1))
        p_xt1_xt = torch.divide(cnts_xt1_xt, (self._len_time - 1))
        p_xt_yt = torch.divide(cnts_xt_yt, (self._len_time - 1))
        p_xt = torch.divide(cnts_xt, (self._len_time - 1))

        numer = torch.multiply(p_xt1_xt_yt, p_xt)
        denom = torch.multiply(p_xt1_xt, p_xt_yt)
        fraction = torch.divide(numer, denom)
        log_val = torch.log2(fraction)
        entropies = torch.multiply(p_xt1_xt_yt, log_val)

        uvals_tot, n_subuvals_tot = torch.unique(uvals_xt1_xt_yt[:, 0], return_counts=True)
        final_bins = torch.repeat_interleave(uvals_tot, n_subuvals_tot)

        entropy_final = torch.bincount(final_bins.to(torch.int32), weights=entropies)

        return entropy_final, pairs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        arr, pairs = batch
        ef, pairs = self(arr, pairs)
        self._batch_predict_ef.append(ef)
        self._batch_predict_pairs.append(pairs)
        return ef, pairs

    def on_predict_epoch_end(self):
        epoch_predict_ef = torch.cat(self._batch_predict_ef, dim=0)
        epoch_predict_ef = self.all_gather(epoch_predict_ef)

        epoch_predict_pairs = torch.cat(self._batch_predict_pairs, dim=0)
        epoch_predict_pairs = self.all_gather(epoch_predict_pairs)

        if self.trainer.is_global_zero:
            for i in range(len(epoch_predict_ef)):
                entropy_final = epoch_predict_ef[i].detach().cpu().numpy()
                pairs = epoch_predict_pairs[i].detach().cpu().numpy()

                self._result_matrix[pairs[:, 0], pairs[:, 1]] = entropy_final

    def return_result(self):
        return self._result_matrix

if __name__ == "__main__":
    x = np.randint(-10, 10, (16, 200), dtype=np.int16)  # (B, C, N)

    pairs = np.array([[0, 1],
                      [0, 2],
                      [3, 4]])
    print(x.shape)
    print(pairs.shape)
    
    model = TELightning()

    print(model)
    t_beg = time.time()
    y, pairs = model(x, pairs)

    t_end = time.time()
    # print(y)
    print(y)
    print("Time elapsed:", t_end - t_beg)
    #
    # from pytorch_model_summary import summary
    #
    # print(summary(model, x, show_input=False))
    #
    # from torchinfo import summary
    #
    # summary(model)