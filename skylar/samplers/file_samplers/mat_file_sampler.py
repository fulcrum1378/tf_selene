import h5py
import numpy as np
import scipy.io

from .file_sampler import FileSampler


def _load_mat_file(filepath, sequence_key, targets_key=None):
    try:  # see if we can load the file using scipy first
        mat = scipy.io.loadmat(filepath)
        targets = None
        if targets_key:
            targets = mat[targets_key]
        return mat[sequence_key], targets
    except (NotImplementedError, ValueError):
        mat = h5py.File(filepath, 'r')
        sequences = mat[sequence_key]
        targets = None
        if targets_key:
            targets = mat[targets_key]
        return sequences, targets, mat


class MatFileSampler(FileSampler):
    def __init__(self,
                 filepath,
                 sequence_key,
                 targets_key=None,
                 random_seed=436,
                 shuffle=True,
                 sequence_batch_axis=0,
                 sequence_alphabet_axis=1,
                 targets_batch_axis=0):
        super(MatFileSampler, self).__init__()
        out = _load_mat_file(
            filepath,
            sequence_key,
            targets_key=targets_key)
        self._sample_seqs = out[0]
        self._sample_tgts = out[1]
        self._mat_fh = None
        if len(out) > 2:
            self._mat_fh = out[2]
        self._seq_batch_axis = sequence_batch_axis
        self._seq_alphabet_axis = sequence_alphabet_axis
        self._seq_final_axis = 3 - sequence_batch_axis - sequence_alphabet_axis
        if self._sample_tgts is not None:
            self._tgts_batch_axis = targets_batch_axis
        self.n_samples = self._sample_seqs.shape[self._seq_batch_axis]

        self._sample_indices = np.arange(
            self.n_samples).tolist()
        self._sample_next = 0

        self._shuffle = shuffle
        if self._shuffle:
            np.random.shuffle(self._sample_indices)

    def sample(self, batch_size=1):
        sample_up_to = self._sample_next + batch_size
        use_indices = None
        if sample_up_to >= len(self._sample_indices):
            if self._shuffle:
                np.random.shuffle(self._sample_indices)
            self._sample_next = 0
            use_indices = self._sample_indices[:batch_size]
        else:
            use_indices = self._sample_indices[self._sample_next:sample_up_to]
        self._sample_next += batch_size
        use_indices = sorted(use_indices)
        if self._seq_batch_axis == 0:
            sequences = self._sample_seqs[use_indices, :, :].astype(float)
        elif self._seq_batch_axis == 1:
            sequences = self._sample_seqs[:, use_indices, :].astype(float)
        else:
            sequences = self._sample_seqs[:, :, use_indices].astype(float)

        if self._seq_batch_axis != 0 or self._seq_alphabet_axis != 2:
            sequences = np.transpose(
                sequences, (self._seq_batch_axis,
                            self._seq_final_axis,
                            self._seq_alphabet_axis))
        if self._sample_tgts is not None:
            if self._tgts_batch_axis == 0:
                targets = self._sample_tgts[use_indices, :].astype(float)
            else:
                targets = self._sample_tgts[:, use_indices].astype(float)
                targets = np.transpose(
                    targets, (1, 0))
            return sequences, targets
        return sequences,

    def get_data(self, batch_size, n_samples=None):
        if not n_samples:
            n_samples = self.n_samples
        sequences = []

        count = batch_size
        while count < n_samples:
            seqs, = self.sample(batch_size=batch_size)
            sequences.append(seqs)
            count += batch_size
        remainder = batch_size - (count - n_samples)
        seqs, = self.sample(batch_size=remainder)
        sequences.append(seqs)
        return sequences

    def get_data_and_targets(self, batch_size, n_samples=None):
        if self._sample_tgts is None:
            raise ValueError(
                "No targets matrix was specified during sampler "
                "initialization. Please use `get_data` instead.")
        if not n_samples:
            n_samples = self.n_samples
        sequences_and_targets = []
        targets_mat = []

        count = batch_size
        while count < n_samples:
            seqs, tgts = self.sample(batch_size=batch_size)
            sequences_and_targets.append((seqs, tgts))
            targets_mat.append(tgts)
            count += batch_size
        remainder = batch_size - (count - n_samples)
        seqs, tgts = self.sample(batch_size=remainder)
        sequences_and_targets.append((seqs, tgts))
        targets_mat.append(tgts)
        # TODO: should not assume targets are always integers
        targets_mat = np.vstack(targets_mat).astype(float)
        return sequences_and_targets, targets_mat
