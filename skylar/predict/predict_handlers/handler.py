from abc import ABCMeta
from abc import abstractmethod
import os
from sys import getsizeof

import h5py


def write_to_tsv_file(data_across_features, info_cols, output_filepath):
    with open(output_filepath, 'a') as output_handle:
        for info_batch, preds_batch in zip(info_cols, data_across_features):
            for info, preds in zip(info_batch, preds_batch):
                preds_str = '\t'.join(
                    probabilities_to_string(list(preds)))
                info_str = '\t'.join([str(i) for i in info])
                output_handle.write("{0}\t{1}\n".format(info_str, preds_str))


def write_to_hdf5_file(data_across_features,
                       info_cols,
                       hdf5_filepath,
                       start_index,
                       info_filepath=None):
    if info_filepath is not None:
        with open(info_filepath, 'a') as info_handle:
            for info_batch in info_cols:
                for info in info_batch:
                    info_str = '\t'.join([str(i) for i in info])
                    info_handle.write("{0}\n".format(info_str))
    with h5py.File(hdf5_filepath, 'a') as hdf5_handle:
        data = hdf5_handle["data"]
        for data_batch in data_across_features:
            data[start_index:(start_index + data_batch.shape[0])] = data_batch
            start_index = start_index + data_batch.shape[0]

    return start_index


def probabilities_to_string(probabilities):
    return ["{:.2e}".format(p) for p in probabilities]


class PredictionsHandler(metaclass=ABCMeta):
    def __init__(self,
                 features,
                 columns_for_ids,
                 output_path_prefix,
                 output_format,
                 output_size=None,
                 write_mem_limit=1500,
                 write_labels=True):
        self.needs_base_pred = False
        self._results = []
        self._samples = []

        self._features = features
        self._columns_for_ids = columns_for_ids
        self._output_path_prefix = output_path_prefix
        self._output_format = output_format
        self._output_size = output_size
        if output_format == 'hdf5' and output_size is None:
            raise ValueError("`output_size` must be specified when "
                             "`output_format` is 'hdf5'.")

        self._output_filepath = None
        self._labels_filepath = None
        self._hdf5_start_index = None

        self._write_mem_limit = write_mem_limit
        self._write_labels = write_labels

    def _create_write_handler(self, handler_filename):
        filename_prefix = None
        if not os.path.isdir(self._output_path_prefix):
            output_path, filename_prefix = os.path.split(
                self._output_path_prefix)
        else:
            output_path = self._output_path_prefix
        if filename_prefix is not None:
            handler_filename = "{0}_{1}".format(
                filename_prefix, handler_filename)
        scores_filepath = os.path.join(output_path, handler_filename)
        if self._output_format == "tsv":
            self._output_filepath = "{0}.tsv".format(scores_filepath)
            with open(self._output_filepath, 'w+') as output_handle:
                column_names = self._columns_for_ids + self._features
                output_handle.write("{0}\n".format(
                    '\t'.join(column_names)))
        elif self._output_format == "hdf5":
            self._output_filepath = "{0}.h5".format(scores_filepath)
            with h5py.File(self._output_filepath, 'w') as output_handle:
                output_handle.create_dataset(
                    "data",
                    (self._output_size, len(self._features)),
                    dtype='float64')
            self._hdf5_start_index = 0

            if not self._write_labels:
                return
            labels_filename = "row_labels.txt"
            if filename_prefix is not None:
                # always output same row labels filename
                if filename_prefix[-4:] == '.ref' or \
                        filename_prefix[-4:] == '.alt':
                    filename_prefix = filename_prefix[:-4]
                labels_filename = "{0}_{1}".format(
                    filename_prefix, labels_filename)
            self._labels_filepath = os.path.join(output_path, labels_filename)
            # create the file
            label_handle = open(self._labels_filepath, 'w+')
            label_handle.write("{0}\n".format(
                '\t'.join(self._columns_for_ids)))

    def _reached_mem_limit(self):
        mem_used = (self._results[0].nbytes * len(self._results) +
                    getsizeof(self._samples[0]) * len(self._samples))
        return mem_used / 10 ** 6 >= self._write_mem_limit

    @abstractmethod
    def handle_batch_predictions(self, *args, **kwargs):
        raise NotImplementedError

    def write_to_file(self):
        if not self._results:
            return None
        if self._hdf5_start_index is not None:
            self._hdf5_start_index = write_to_hdf5_file(
                self._results,
                self._samples,
                self._output_filepath,
                self._hdf5_start_index,
                info_filepath=self._labels_filepath)
        else:
            write_to_tsv_file(self._results,
                              self._samples,
                              self._output_filepath)
        self._results = []
        self._samples = []
