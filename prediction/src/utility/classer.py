import logging
import torch
from torch import nn
import math
import numpy as np
from typing import Optional, Iterator
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
from collections import defaultdict
from utility.function import mkdir


""" BucketSampler """
class BucketSampler(Sampler):
    def __init__(self,
                 dataset: Dataset,
                 rc_lens_list,
                 batch_size,
                 is_dist: bool = True,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0) -> None:
        if is_dist:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval"
                    " [0, {}]".format(rank, num_replicas - 1))
        else:
            num_replicas = 1
            rank = 0

        self.num_samples = 0
        self.dataset = dataset
        self.rc_lens_list = rc_lens_list
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

    def __iter__(self) -> Iterator:

        '''
        separate indices into <num_replicas> ranks
        '''
        indices_all = list(range(len(self.dataset)))  # all relation category train edges index

        n_rc = len(self.rc_lens_list)  # all relation category count
        rc_idx_list = list(range(n_rc))  # relation category index [0, 1, 2, 3, ...]
        rc_btn_list = [0]*n_rc  # relation category batch num [0, 0, 0, 0, ...]

        i = 0
        indices_dict = defaultdict(list)
        while True:
            np.random.seed(self.seed + self.epoch)
            ### chose relation category randomly
            cu_rc_idx = np.random.choice(rc_idx_list)  # current relation category index

            rc_s = sum(self.rc_lens_list[:cu_rc_idx])  # relation category train edges start index
            rc_e = rc_s + self.rc_lens_list[cu_rc_idx]  # relation category train edges end index
            rc_indices = indices_all[rc_s:rc_e]

            bt_s = rc_btn_list[cu_rc_idx] * self.batch_size  # relation category batch index start
            bt_e = bt_s + self.batch_size  # relation category batch index end
            bt_indices = rc_indices[bt_s:bt_e]
            rc_btn_list[cu_rc_idx] += 1  # update relation category batch num

            ### rank id
            rank = i % self.num_replicas
            ### add indices to the current rank
            indices_dict[rank] += bt_indices
            ### rank id ++
            i += 1

            ### if current relation category is not enough for one batch, remove it
            if len(rc_indices) - bt_e < self.batch_size:
                rc_idx_list.remove(cu_rc_idx)

            if len(rc_idx_list) == 0:
                break

        ### subsample for each rank
        indices = indices_dict[self.rank]
        self.num_samples = len(indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


""" HingeLoss """
class HingeLoss(nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()

        self.margin = margin

    def forward(self, pos, neg):
        diff = torch.relu(torch.sub(neg, pos - self.margin))
        loss = torch.sum(diff)
        return loss


""" logger """
def logger(log_file):
    logging.basicConfig(
                    level=logging.DEBUG,
                    format="%(asctime)s - %(name)s : %(levelname)s - %(message)s",
                    datefmt="%y-%m-%d %H:%M:%S",
                    filename=log_file,
                    filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s : %(levelname)s - %(message)s")
    console.setFormatter(formatter)

    logging.getLogger("").addHandler(console)
    # logging.warning("Jackdaws love my big sphinx of quartz.")

    return logging
