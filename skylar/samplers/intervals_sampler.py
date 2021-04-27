from collections import namedtuple
import logging
import random

import numpy as np

from .online_sampler import OnlineSampler
from ..utils import get_indices_and_probabilities

logger = logging.getLogger(__name__)

SampleIndices = namedtuple("SampleIndices", ["indices", "weights"])


class IntervalsSampler(OnlineSampler):
    def __init__(self,
                 reference_sequence,
                 target_path,
                 features,
                 intervals_path,
                 sample_negative=False,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1000,
                 center_bin_to_predict=200,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=["test"],
                 output_dir=None):
        super(IntervalsSampler, self).__init__(
            reference_sequence,
            target_path,
            features,
            seed=seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            feature_thresholds=feature_thresholds,
            mode=mode,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": None, "sample_next": 0}

        self.sample_from_intervals = []
        self.interval_lengths = []

        if self._holdout_type == "chromosome":
            self._partition_dataset_chromosome(intervals_path)
        else:
            self._partition_dataset_proportion(intervals_path)

        for mode in self.modes:
            self._update_randcache(mode=mode)

        self.sample_negative = sample_negative

    def _partition_dataset_proportion(self, intervals_path):
        with open(intervals_path, 'r') as file_handle:
            for line in file_handle:
                cols = line.strip().split('\t')
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])
                self.sample_from_intervals.append((chrom, start, end))
                self.interval_lengths.append(end - start)
        n_intervals = len(self.sample_from_intervals)

        # all indices in the intervals list are shuffled
        select_indices = list(range(n_intervals))
        np.random.shuffle(select_indices)

        # the first section of indices is used as the validation set
        n_indices_validate = int(n_intervals * self.validation_holdout)
        val_indices, val_weights = get_indices_and_probabilities(
            self.interval_lengths, select_indices[:n_indices_validate])
        self._sample_from_mode["validate"] = SampleIndices(
            val_indices, val_weights)

        if self.test_holdout:
            # if applicable, the second section of indices is used as the
            # test set
            n_indices_test = int(n_intervals * self.test_holdout)
            test_indices_end = n_indices_test + n_indices_validate
            test_indices, test_weights = get_indices_and_probabilities(
                self.interval_lengths,
                select_indices[n_indices_validate:test_indices_end])
            self._sample_from_mode["test"] = SampleIndices(
                test_indices, test_weights)

            # remaining indices are for the training set
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[test_indices_end:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)
        else:
            # remaining indices are for the training set
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[n_indices_validate:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)

    def _partition_dataset_chromosome(self, intervals_path):
        for mode in self.modes:
            self._sample_from_mode[mode] = SampleIndices([], [])
        with open(intervals_path, 'r') as file_handle:
            for index, line in enumerate(file_handle):
                cols = line.strip().split('\t')
                chrom = cols[0]
                start = int(cols[1])
                end = int(cols[2])
                if chrom in self.validation_holdout:
                    self._sample_from_mode["validate"].indices.append(
                        index)
                elif self.test_holdout and chrom in self.test_holdout:
                    self._sample_from_mode["test"].indices.append(
                        index)
                else:
                    self._sample_from_mode["train"].indices.append(
                        index)
                self.sample_from_intervals.append((chrom, start, end))
                self.interval_lengths.append(end - start)

        for mode in self.modes:
            sample_indices = self._sample_from_mode[mode].indices
            indices, weights = get_indices_and_probabilities(
                self.interval_lengths, sample_indices)
            self._sample_from_mode[mode] = \
                self._sample_from_mode[mode]._replace(
                    indices=indices, weights=weights)

    def _retrieve(self, chrom, position):
        bin_start = position - self._start_radius
        bin_end = position + self._end_radius
        retrieved_targets = self.target.get_feature_data(
            chrom, bin_start, bin_end)
        if not self.sample_negative and np.sum(retrieved_targets) == 0:
            logger.info("No features found in region surrounding "
                        "region \"{0}\" position {1}. Sampling again.".format(
                chrom, position))
            return None

        window_start = position - self._start_window_radius
        window_end = position + self._end_window_radius
        strand = self.STRAND_SIDES[random.randint(0, 1)]
        retrieved_seq = \
            self.reference_sequence.get_encoding_from_coords(
                chrom, window_start, window_end, strand)
        if retrieved_seq.shape[0] == 0:
            logger.info("Full sequence centered at region \"{0}\" position "
                        "{1} could not be retrieved. Sampling again.".format(
                chrom, position))
            return None
        elif np.sum(retrieved_seq) / float(retrieved_seq.shape[0]) < 0.60:
            logger.info("Over 30% of the bases in the sequence centered "
                        "at region \"{0}\" position {1} are ambiguous ('N'). "
                        "Sampling again.".format(chrom, position))
            return None

        if self.mode in self._save_datasets:
            feature_indices = ';'.join(
                [str(f) for f in np.nonzero(retrieved_targets)[0]])
            self._save_datasets[self.mode].append(
                [chrom,
                 window_start,
                 window_end,
                 strand,
                 feature_indices])
            if len(self._save_datasets[self.mode]) > 200000:
                self.save_dataset_to_file(self.mode)
        return (retrieved_seq, retrieved_targets)

    def _update_randcache(self, mode=None):
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = np.random.choice(
            self._sample_from_mode[mode].indices,
            size=len(self._sample_from_mode[mode].indices),
            replace=True,
            p=self._sample_from_mode[mode].weights)
        self._randcache[mode]["sample_next"] = 0

    def sample(self, batch_size=1, mode=None):
        mode = mode if mode else self.mode
        sequences = np.zeros((batch_size, self.sequence_length, 4))
        targets = np.zeros((batch_size, self.n_features))
        n_samples_drawn = 0
        while n_samples_drawn < batch_size:
            sample_index = self._randcache[mode]["sample_next"]
            if sample_index == len(self._sample_from_mode[mode].indices):
                self._update_randcache()
                sample_index = 0

            rand_interval_index = \
                self._randcache[mode]["cache_indices"][sample_index]
            self._randcache[mode]["sample_next"] += 1

            interval_info = self.sample_from_intervals[rand_interval_index]
            interval_length = self.interval_lengths[rand_interval_index]

            chrom = interval_info[0]
            position = int(
                interval_info[1] + random.uniform(0, 1) * interval_length)

            retrieve_output = self._retrieve(chrom, position)
            if not retrieve_output:
                continue
            seq, seq_targets = retrieve_output
            sequences[n_samples_drawn, :, :] = seq
            targets[n_samples_drawn, :] = seq_targets
            n_samples_drawn += 1
        return sequences, targets
