import numpy as np

from .file_sampler import FileSampler


class BedFileSampler(FileSampler):
    def __init__(self,
                 filepath,
                 reference_sequence,
                 n_samples,
                 sequence_length=None,
                 targets_avail=False,
                 n_features=None):
        super(BedFileSampler, self).__init__()
        self.filepath = filepath
        self._file_handle = open(self.filepath, 'r')
        self.reference_sequence = reference_sequence
        self.sequence_length = sequence_length
        self.targets_avail = targets_avail
        self.n_features = n_features
        self.n_samples = n_samples

    def sample(self, batch_size=1):
        sequences = []
        targets = None
        if self.targets_avail:
            targets = []
        while len(sequences) < batch_size:
            line = self._file_handle.readline()
            if not line:
                # reaches the end of the file.
                self._file_handle.close()
                self._file_handle = open(self.filepath, 'r')
                line = self._file_handle.readline()
            cols = line.split('\t')
            chrom = cols[0]
            start = int(cols[1])
            end = int(cols[2])
            strand_side = None
            features = None

            if len(cols) == 5:
                strand_side = cols[3]
                features = cols[4].strip()
            elif len(cols) == 4 and self.targets_avail:
                features = cols[3].strip()
            elif len(cols) == 4:
                strand_side = cols[3].strip()

            # if strand_side is None, assume strandedness does not matter.
            # can change this to randomly selecting +/- later
            strand_side = '+'
            n = end - start
            if self.sequence_length and n < self.sequence_length:
                diff = (self.sequence_length - n) / 2
                pad_l = int(np.floor(diff))
                pad_r = int(np.ceil(diff))
                start = start - pad_l
                end = end + pad_r
            elif self.sequence_length and n > self.sequence_length:
                start = int((n - self.sequence_length) // 2)
                end = int(start + self.sequence_length)

            sequence = self.reference_sequence.get_encoding_from_coords(
                chrom, start, end, strand=strand_side)
            if sequence.shape[0] == 0:
                continue

            sequences.append(sequence)
            if self.targets_avail:
                tgts = np.zeros((self.n_features))
                features = [int(f) for f in features.split(';') if f]
                tgts[features] = 1
                targets.append(tgts.astype(float))

        sequences = np.array(sequences)
        if self.targets_avail:
            targets = np.array(targets)
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
        if not self.targets_avail:
            raise ValueError(
                "No targets are specified in the *.bed file. "
                "Please use `get_data` instead.")
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
        targets_mat = np.vstack(targets_mat).astype(int)
        return sequences_and_targets, targets_mat
