import time

import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl
from scipy.stats import norm
from tqdm import tqdm


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
    def __init__(self,
                 arr=None,
                 len_time=None,
                 dt=1,
                 n_bins=None,
                 surrogate=False,
                 num_surrogate=1000,
                 threshold=0.05,
                 ):
        super().__init__()
        self._arr = arr
        self._len_time = len_time
        self._dt = dt
        self._n_bins = n_bins

        self._result_matrix = np.zeros((len(arr), len(arr)), dtype=np.float32)

        self._batch_predict_ef = []
        self._batch_predict_pairs = []

        self._surrogate = surrogate
        self._num_surr = num_surrogate
        self._threshold = threshold

    def compute_te(self,
                   arr=None,
                   t_pairs=None,
                   s_pairs=None,
                   tile_inds_pair=None,
                   ):

        target_arr = torch.index_select(arr, 0, t_pairs)
        source_arr = torch.index_select(arr, 0, s_pairs)
        vals = torch.stack((target_arr[:, self._dt:, :],
                            target_arr[:, :-self._dt, :],
                            source_arr[:, :-self._dt, :]),
                           dim=3)

        t_vals = torch.permute(vals, dims=(2, 0, 1, 3))

        pair_vals = torch.concatenate((tile_inds_pair[:, None], torch.reshape(t_vals, (-1, 3))), dim=1)

        # n_bins = torch.tensor(self._n_bins, dtype=torch.int32, device=self.device)
        # n_bins = torch.index_select(n_bins, 0, t_pairs)
        # n_bins = torch.repeat_interleave(n_bins, (self._len_time - 1))
        # n_bins = torch.tile(n_bins, (arr.shape[-1],))
        #
        # left_bools = torch.tensor(
        #     torch.logical_and(
        #         torch.greater_equal(pair_vals[:, 2], 0),
        #         torch.less(pair_vals[:, 2], n_bins)
        #     )
        # )
        # left_inds = torch.where(left_bools)[0]
        #
        # pair_vals = torch.index_select(pair_vals, 0, left_inds)

        uvals_xt1_xt_yt, cnts_xt1_xt_yt = torch.unique(pair_vals, return_counts=True, dim=0)
        uvals_xt1_xt, cnts_xt1_xt = torch.unique(pair_vals[:, :-1], return_counts=True, dim=0)
        uvals_xt_yt, cnts_xt_yt = torch.unique(
            torch.index_select(pair_vals, 1, torch.tensor([0, 2, 3], device=self.device)),
            return_counts=True, dim=0)
        uvals_xt, cnts_xt = torch.unique(torch.index_select(pair_vals, 1, torch.tensor([0, 2], device=self.device)),
                                         return_counts=True, dim=0)

        subuvals_xt1_xt, n_subuvals_xt1_xt = torch.unique(uvals_xt1_xt_yt[:, :-1], return_counts=True, dim=0)
        subuvals_xt_yt, n_subuvals_xt_yt = torch.unique(
            torch.index_select(uvals_xt1_xt_yt, 1, torch.tensor([0, 2, 3], device=self.device)),
            return_counts=True, dim=0)
        subuvals_xt, n_subuvals_xt = torch.unique(
            torch.index_select(uvals_xt1_xt_yt, 1, torch.tensor([0, 2], device=self.device)),
            return_counts=True, dim=0)

        cnts_xt1_xt = torch.repeat_interleave(cnts_xt1_xt, n_subuvals_xt1_xt)

        cnts_xt_yt = torch.repeat_interleave(cnts_xt_yt, n_subuvals_xt_yt)
        ind_xt_yt = lexsort(torch.index_select(uvals_xt1_xt_yt, 1, torch.tensor([3, 2, 0], device=self.device)).T)
        ind2ori_xt_yt = torch.argsort(ind_xt_yt)
        cnts_xt_yt = torch.take(cnts_xt_yt, ind2ori_xt_yt)

        cnts_xt = torch.repeat_interleave(cnts_xt, n_subuvals_xt)
        ind_xt = lexsort(torch.index_select(uvals_xt1_xt_yt, 1, torch.tensor([2, 0], device=self.device)).T)
        ind2ori_xt = torch.argsort(ind_xt)
        cnts_xt = torch.take(cnts_xt, ind2ori_xt)

        p_xt1_xt_yt = torch.divide(cnts_xt1_xt_yt, (self._len_time - 1) * arr.shape[-1])

        numer = torch.multiply(cnts_xt1_xt_yt, cnts_xt)
        denom = torch.multiply(cnts_xt1_xt, cnts_xt_yt)
        fraction = torch.divide(numer, denom)
        log_val = torch.log2(fraction)
        entropies = torch.multiply(p_xt1_xt_yt, log_val)

        uvals_tot, n_subuvals_tot = torch.unique(uvals_xt1_xt_yt[:, 0], return_counts=True)
        final_bins = torch.repeat_interleave(uvals_tot, n_subuvals_tot)

        entropy_final = torch.bincount(final_bins.to(torch.int32), weights=entropies)

        return entropy_final

    def forward(self, pairs):
        arr = self._arr

        arr = torch.tensor(arr, dtype=torch.float32, device=self.device)
        pairs = torch.tensor(pairs, dtype=torch.int32, device=self.device)

        if self._len_time is None:
            self._len_time = arr.shape[1]

        inds_pair = torch.arange(len(pairs)).to(self.device)

        tile_inds_pair = torch.repeat_interleave(inds_pair, self._len_time - 1)
        tile_inds_pair = torch.tile(tile_inds_pair, (arr.shape[-1],))

        t_pairs = pairs[:, 0]
        s_pairs = pairs[:, 1]

        entropy_final = self.compute_te(arr=arr,
                                        t_pairs=t_pairs,
                                        s_pairs=s_pairs,
                                        tile_inds_pair=tile_inds_pair
                                        )

        if self._surrogate is True:
            surrogate_tes = []
            for i in tqdm(range(self._num_surr)):
                idx = np.random.rand(*arr.shape).argsort(axis=1)
                arr = torch.take_along_dim(arr, torch.tensor(idx).to(self.device), dim=1)

                entropy_surrogate = self.compute_te(arr=arr,
                                                    t_pairs=t_pairs,
                                                    s_pairs=s_pairs,
                                                    tile_inds_pair=tile_inds_pair)

                entropy_surrogate = entropy_surrogate.detach().cpu().numpy()

                surrogate_tes.append(entropy_surrogate)

            surrogate_tes = np.array(surrogate_tes)

            # # 2안: surrogate의 각 TE로부터 분포 구성 -> 상위 k% 값 추출
            means = np.mean(surrogate_tes, axis=0)
            std = np.std(surrogate_tes, axis=0)
            top_values = norm.ppf((1 - self._threshold), loc=means, scale=std)

            # original te 값과 비교 -> original te < surrogate top te 이면 0
            entropy_final[entropy_final <= torch.tensor(top_values).to(self.device)] = 0.0

        return entropy_final, pairs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pairs = batch
        # print(pairs.shape)
        # print(pairs)
        ef, pairs = self(pairs)
        self._batch_predict_ef.append(ef)
        self._batch_predict_pairs.append(pairs)
        return ef, pairs

    def on_predict_epoch_end(self):
        epoch_predict_ef = torch.cat(self._batch_predict_ef, dim=0)
        epoch_predict_ef = self.all_gather(epoch_predict_ef)

        epoch_predict_pairs = torch.cat(self._batch_predict_pairs, dim=0)
        epoch_predict_pairs = self.all_gather(epoch_predict_pairs)

        if self.trainer.is_global_zero:
            if epoch_predict_pairs.dim() > 2:
                for i in range(len(epoch_predict_ef)):
                    entropy_final = epoch_predict_ef[i].detach().cpu().numpy()
                    pairs = epoch_predict_pairs[i].detach().cpu().numpy()

                    self._result_matrix[pairs[:, 0], pairs[:, 1]] = entropy_final
            else:
                entropy_final = epoch_predict_ef.detach().cpu().numpy()
                pairs = epoch_predict_pairs.detach().cpu().numpy()
                self._result_matrix[pairs[:, 0], pairs[:, 1]] = entropy_final

    def return_result(self):
        return self._result_matrix

if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    from mate.dataset import PairDataSet

    x = np.random.randint(-10, 10, (16, 200), dtype=np.int16)  # (B, C, N)

    stds = np.std(x, axis=1, ddof=1)
    mins = np.min(x, axis=1)
    maxs = np.max(x, axis=1)

    n_bins = np.ceil((maxs - mins) / stds).T.astype(dtype)

    pairs = np.array([[0, 1],
                      [0, 2],
                      [3, 4]])
    print(x.shape)
    print(pairs.shape)
    
    model = TELightning(arr=x, len_time=200, dt=1, n_bins=n_bins)

    print(model)

    def custom_collate(batch):
        n_devices = None

        pairs = [item[1] for item in batch]
        # arr = batch[0][0]

        return np.stack(pairs)

    dset_pair = PairDataSet(arr=x, pairs=pairs)

    dloader_pair = DataLoader(dset_pair,
                              batch_size=32,
                              shuffle=False,
                              num_workers=0,
                              collate_fn=custom_collate)

    trainer = L.Trainer(accelerator='gpu',
                        devices=1,
                        num_nodes=1,
                        strategy="auto")

    trainer.predict(model, dloader_pair)

    # from pytorch_model_summary import summary
    #
    # print(summary(model, x, show_input=False))
    #
    # from torchinfo import summary
    #
    # summary(model)