from abc import ABCMeta
import os
import random

import numpy as np

from .sampler import Sampler
from ..targets import GenomicFeatures


class OnlineSampler(Sampler, metaclass=ABCMeta):
    STRAND_SIDES = ('+', '-')
    """
    Defines the strands that features can be sampled from.
    """

    def __init__(self,
                 reference_sequence,
                 target_path,
                 features,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1001,
                 center_bin_to_predict=201,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=[],
                 output_dir=None):
        super(OnlineSampler, self).__init__(
            features,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed + 1)

        if isinstance(center_bin_to_predict, int):
            if (sequence_length + center_bin_to_predict) % 2 != 0:
                raise ValueError(
                    "Sequence length of {0} with a center bin length of {1} "
                    "is invalid. These 2 inputs should both be odd or both be "
                    "even.".format(sequence_length, center_bin_to_predict))

        # specifying a test holdout partition is optional
        if test_holdout:
            self.modes.append("test")
            if isinstance(validation_holdout, (list,)) and \
                    isinstance(test_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
                self.test_holdout = [str(c) for c in test_holdout]
                self._holdout_type = "chromosome"
            elif isinstance(validation_holdout, float) and \
                    isinstance(test_holdout, float):
                self.validation_holdout = validation_holdout
                self.test_holdout = test_holdout
                self._holdout_type = "proportion"
            else:
                raise ValueError(
                    "Validation holdout and test holdout must have the "
                    "same type (list or float) but validation was "
                    "type {0} and test was type {1}".format(
                        type(validation_holdout), type(test_holdout)))
        else:
            self.test_holdout = None
            if isinstance(validation_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
                self._holdout_type = "chromosome"
            elif isinstance(validation_holdout, float):
                self.validation_holdout = validation_holdout
                self._holdout_type = "proportion"
            else:
                raise ValueError(
                    "Validation holdout must be of type list (chromosomal "
                    "holdout) or float (proportion holdout) but was type "
                    "{0}.".format(type(validation_holdout)))

        if mode not in self.modes:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.modes, mode))
        self.mode = mode

        self.sequence_length = sequence_length
        window_radius = int(self.sequence_length / 2)
        self._start_window_radius = window_radius
        self._end_window_radius = window_radius
        if self.sequence_length % 2 != 0:
            self._end_window_radius += 1

        if isinstance(center_bin_to_predict, int):
            bin_radius = int(center_bin_to_predict / 2)
            self._start_radius = bin_radius
            self._end_radius = bin_radius
            if center_bin_to_predict % 2 != 0:
                self._end_radius += 1
        else:
            if not isinstance(center_bin_to_predict, list) or \
                    len(center_bin_to_predict) != 2:
                raise ValueError(
                    "`center_bin_to_predict` needs to be either an int or a list of "
                    "two ints, but was type '{0}'".format(
                        type(center_bin_to_predict)))
            else:
                bin_start, bin_end = center_bin_to_predict
                if bin_start < 0 or bin_end > self.sequence_length:
                    ValueError(
                        "center_bin_to_predict [{0}, {1}]"
                        "is out-of-bound for sequence length {3}.".format(
                            bin_start, bin_end, self.sequence_length))
                self._start_radius = self._start_window_radius - bin_start
                self._end_radius = self._end_window_radius - (self.sequence_length - bin_end)

        self.reference_sequence = reference_sequence
        self.n_features = len(self._features)
        self.target = GenomicFeatures(
            target_path, self._features,
            feature_thresholds=feature_thresholds)
        self._save_filehandles = {}

    def get_feature_from_index(self, index):
        return self.target.index_feature_dict[index]

    def get_sequence_from_encoding(self, encoding):
        return self.reference_sequence.encoding_to_sequence(encoding)

    def save_dataset_to_file(self, mode, close_filehandle=False):
        if mode not in self._save_datasets:
            return
        samples = self._save_datasets[mode]
        if mode not in self._save_filehandles:
            self._save_filehandles[mode] = open(
                os.path.join(self._output_dir,
                             "{0}_data.bed".format(mode)),
                'w+')
        file_handle = self._save_filehandles[mode]
        while len(samples) > 0:
            cols = samples.pop(0)
            line = '\t'.join([str(c) for c in cols])
            file_handle.write("{0}\n".format(line))
        if close_filehandle:
            file_handle.close()

    def get_data_and_targets(self, batch_size, n_samples=None, mode=None):
        if mode is not None:
            self.set_mode(mode)
        else:
            mode = self.mode
        sequences_and_targets = []
        if n_samples is None and mode == "validate":
            n_samples = 32000
        elif n_samples is None and mode == "test":
            n_samples = 640000

        n_batches = int(n_samples / batch_size)
        for _ in range(n_batches):
            inputs, targets = self.sample(batch_size)
            sequences_and_targets.append((inputs, targets))
        targets_mat = np.vstack([t for (s, t) in sequences_and_targets])
        if mode in self._save_datasets:
            self.save_dataset_to_file(mode, close_filehandle=True)
        return sequences_and_targets, targets_mat

    def get_dataset_in_batches(self, mode, batch_size, n_samples=None):
        return self.get_data_and_targets(
            batch_size, n_samples=n_samples, mode=mode)

    def get_validation_set(self, batch_size, n_samples=None):
        return self.get_dataset_in_batches(
            "validate", batch_size, n_samples=n_samples)

    def get_test_set(self, batch_size, n_samples=None):
        if "test" not in self.modes:
            raise ValueError("No test partition of the data was specified "
                             "during initialization. Cannot use method "
                             "`get_test_set`.")
        return self.get_dataset_in_batches("test", batch_size, n_samples)
