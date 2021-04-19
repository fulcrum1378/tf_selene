import sys

from functools import wraps
import h5py
import numpy as np
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


class _SamplerDataset(tf.data.Dataset):
    def __init__(self, sampler):
        super(_SamplerDataset, self).__init__()
        self.sampler = sampler

    def __getitem__(self, index):
        sequences, targets = self.sampler.sample(
            batch_size=1 if isinstance(index, int) else len(index))
        if sequences.shape[0] == 1:
            sequences = sequences[0, :]
            targets = targets[0, :]
        return sequences, targets

    def __len__(self):
        return sys.maxsize


class SamplerDataLoader(DataLoader):
    def __init__(self,
                 sampler,
                 num_workers=1,
                 batch_size=1,
                 seed=436):
        def worker_init_fn(worker_id):
            np.random.seed(seed + worker_id)

        args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "worker_init_fn": worker_init_fn
        }

        super(SamplerDataLoader, self).__init__(_SamplerDataset(sampler), **args)
        self.seed = seed


class _H5Dataset(tf.data.Dataset):
    def __init__(self,
                 file_path,
                 in_memory=False,
                 unpackbits=False,
                 sequence_key="sequences",
                 targets_key="targets"):
        super(_H5Dataset, self).__init__()
        self.file_path = file_path
        self.in_memory = in_memory
        self.unpackbits = unpackbits

        self._initialized = False
        self._sequence_key = sequence_key
        self._targets_key = targets_key

    def init(func):
        # delay initialization to allow multiprocessing
        @wraps(func)
        def dfunc(self, *args, **kwargs):
            if not self.initialized:
                self.db = h5py.File(self.file_path, 'r')
                if self.unpackbits:
                    self.s_len = self.db['{0}_length'.format(self._sequence_key)][()]
                    self.t_len = self.db['{0}_length'.format(self._targets_key)][()]
                if self.in_memory:
                    self.sequences = np.asarray(self.db[self._sequence_key])
                    self.targets = np.asarray(self.db[self._targets_key])
                else:
                    self.sequences = self.db[self._sequence_key]
                    self.targets = self.db[self._targets_key]
                self.initialized = True
            return func(self, *args, **kwargs)

        return dfunc

    @init
    def __getitem__(self, index):
        if isinstance(index, int):
            index = index % self.sequences.shape[0]
        sequence = self.sequences[index, :, :]
        targets = self.targets[index, :]
        if self.unpackbits:
            sequence = np.unpackbits(sequence, axis=-2)
            nulls = np.sum(sequence, axis=-1) == sequence.shape[-1]
            sequence = sequence.astype(float)
            sequence[nulls, :] = 1.0 / sequence.shape[-1]
            targets = np.unpackbits(
                targets, axis=-1).astype(float)
            if sequence.ndim == 3:
                sequence = sequence[:, :self.s_len, :]
            else:
                sequence = sequence[:self.s_len, :]
            if targets.ndim == 2:
                targets = targets[:, :self.t_len]
            else:
                targets = targets[:self.t_len]
        return (tf.convert_to_tensor(sequence.astype(np.float32)),
                tf.convert_to_tensor(targets.astype(np.float32)))

    @init
    def __len__(self):
        return self.sequences.shape[0]


class H5DataLoader(DataLoader):
    def __init__(self,
                 filepath,
                 in_memory=False,
                 num_workers=1,
                 use_subset=None,
                 batch_size=1,
                 shuffle=True,
                 unpackbits=False,
                 sequence_key="sequences",
                 targets_key="targets"):
        args = {
            "batch_size": batch_size,
            "num_workers": 0 if in_memory else num_workers,
            "pin_memory": True
        }
        if use_subset is not None:
            if isinstance(use_subset, int):
                use_subset = list(range(use_subset))
            elif isinstance(use_subset, tuple) and len(use_subset) == 2:
                use_subset = list(range(use_subset[0], use_subset[1]))
            args["sampler"] = SubsetRandomSampler(use_subset)
        else:
            args["shuffle"] = shuffle
        super(H5DataLoader, self).__init__(
            _H5Dataset(filepath,
                       in_memory=in_memory,
                       unpackbits=unpackbits,
                       sequence_key=sequence_key,
                       targets_key=targets_key),
            **args)
