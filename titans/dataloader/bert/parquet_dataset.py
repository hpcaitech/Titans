import json
import numpy as np
import os
import random
import torch

import torch.distributed as dist
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from lddl.types import File
from lddl.utils import get_num_samples_of_parquet
from lddl.random import sample
from lddl.torch.datasets import ShuffleBuffer
from lddl.torch.datasets import ParquetDataset as PD


class ParquetDataset(PD, IterableDataset):

  def __init__(
      self,
      file_paths,
      transform=lambda x: x,
      shuffle_buffer_size=16384,
      shuffle_buffer_warmup_factor=16,
      base_seed=12345,
      logger=None,
      start_epoch=0,
      process_group=None
  ):
    # we do not want to init the original PD as it is overridden by this function
    # we only init with IterabledDataset
    IterableDataset.__init__(self)
    self._transform = transform
    self._shuffle_buffer_size = shuffle_buffer_size
    self._shuffle_buffer_warmup_factor = shuffle_buffer_warmup_factor
    self._base_seed = base_seed

    self._rank = dist.get_rank(group=process_group)
    self._world_size = dist.get_world_size(group=process_group)
    self._process_group = process_group

    self._epoch = start_epoch - 1

    self._logger = logger

    assert len(file_paths) % self._world_size == 0
    self._files = self._get_files(file_paths)
    max_num_samples_per_file = max((f.num_samples for f in self._files))
    min_num_samples_per_file = min((f.num_samples for f in self._files))
    assert min_num_samples_per_file + 1 == max_num_samples_per_file
    self._num_samples_per_file = min_num_samples_per_file
    total_num_samples = sum((f.num_samples for f in self._files))
    num_samples_lost = (total_num_samples -
                        self._num_samples_per_file * len(self._files))
    self._logger.to('node').warning('lost {}/{}={}% samples in total'.format(
        num_samples_lost,
        total_num_samples,
        num_samples_lost / total_num_samples * 100,
    ))

    self._world_rng_state = None
    self._worker_rng_state = None
  
  def _get_files(self, file_paths):
    all_files_num_samples = torch.zeros((len(file_paths),), dtype=torch.long)
    if self._world_size > 1 and torch.distributed.get_backend() == 'nccl':
      all_files_num_samples = all_files_num_samples.to('cuda')
    # Figure out how many samples in each file.
    num_samples_cache = {}  # Map dirname to the dict of {basename: num_samples}

    for idx in range(self._rank, len(file_paths), self._world_size):
      fp = file_paths[idx]
      dn = os.path.dirname(fp)
      bn = os.path.basename(fp)
      # Load the num_samples cache file if it exists.
      if dn not in num_samples_cache:
        nsfp = os.path.join(dn, '.num_samples.json')
        try:
          with open(nsfp, 'r') as nsf:
            num_samples_cache[dn] = json.load(nsf)
        except Exception as e:
          self._logger.to('rank').warning('failed to load {}: {}'.format(
              nsfp, e))
          # Mark that the num_samples cache file doesn't exist for this
          # directory.
          num_samples_cache[dn] = None
      if num_samples_cache[dn] is not None and bn in num_samples_cache[dn]:
        all_files_num_samples[idx] = num_samples_cache[dn][bn]
      else:
        # Find out num_samples by loading the parquet table.
        all_files_num_samples[idx] = get_num_samples_of_parquet(fp)
    if self._world_size > 1:
      # Sync. accross all ranks.
      torch.distributed.all_reduce(
          all_files_num_samples,
          op=torch.distributed.ReduceOp.SUM,
          group=self._process_group
      )
    all_files_num_samples = all_files_num_samples.tolist()
    return [File(fp, ns) for fp, ns in zip(file_paths, all_files_num_samples)]

  def __len__(self):
    """ This function only returns how many samples per rank will be yielded
    by this dataset.

    Note that, len(dataloader), where dataloader is a PyTorch DataLoader
    wrapping this dataset, does NOT return the accurate number of batches. This
    is because, when (num_samples_per_file * num_files_per_worker) is not
    divisible by batch_size, each worker is going to generate a partial batch
    at the very end.

    However, PyTorch DataLoader's __len__ only divide the number returned from
    this function by batch_size, which would be smaller than the actual number
    of batches by at most (num_workers - 1).

    We need to patch PyTorch DataLoader function for this function to behave
    correctly.
    """
    return self._num_samples_per_file * len(self._files) // self._world_size

  @property
  def num_samples_per_file(self):
    return self._num_samples_per_file

  @property
  def num_files_per_rank(self):
    return len(self._files) // self._world_size

  def _decode_record_batch(self, b):
    raise NotImplementedError('ParquetDataset is an abstract/interface class!')

  def _world_identical_sample(self, population, k, counts=None):
    s, self._world_rng_state = sample(
        population,
        k,
        rng_state=self._world_rng_state,
    )
    return s

  def _init_worker(self):
    worker_info = get_worker_info()
    if worker_info is None:
      num_workers_per_rank = 1
      worker_rank = 0
    else:
      num_workers_per_rank = worker_info.num_workers
      worker_rank = worker_info.id
    assert (len(self._files) % (self._world_size * num_workers_per_rank) == 0)
    self._logger.init_for_worker(worker_rank)
    return worker_rank, num_workers_per_rank

  def _init_rng_states(self, worker_rank, num_workers_per_rank):
    orig_rng_state = random.getstate()

    random.seed(self._base_seed + self._epoch)
    self._world_rng_state = random.getstate()

    random.seed(self._base_seed +
                (self._epoch * self._world_size + self._rank) *
                num_workers_per_rank + worker_rank)
    self._worker_rng_state = random.getstate()

    random.setstate(orig_rng_state)

  def __iter__(self):
    self._epoch += 1

    worker_rank, num_workers_per_rank = self._init_worker()
    self._init_rng_states(worker_rank, num_workers_per_rank)

    files = self._world_identical_sample(self._files, k=len(self._files))
    rank_files = files[self._rank::self._world_size]
    worker_files = rank_files[worker_rank::num_workers_per_rank]

    sb = ShuffleBuffer(
        worker_files,
        self._num_samples_per_file * len(worker_files),
        lambda b: self._decode_record_batch(b),
        self._shuffle_buffer_size,
        self._shuffle_buffer_warmup_factor,
        self._logger,
        self._worker_rng_state,
    )
    for sample in iter(sb):
      yield self._transform(sample)