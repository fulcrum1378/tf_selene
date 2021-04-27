import os

from .handler import PredictionsHandler
from .write_predictions_handler import WritePredictionsHandler


class WriteRefAltHandler(PredictionsHandler):
    def __init__(self,
                 features,
                 columns_for_ids,
                 output_path_prefix,
                 output_format,
                 output_size=None,
                 write_mem_limit=1500,
                 write_labels=True):
        super(WriteRefAltHandler, self).__init__(
            features,
            columns_for_ids,
            output_path_prefix,
            output_format,
            output_size=output_size,
            write_mem_limit=write_mem_limit,
            write_labels=write_labels)

        self.needs_base_pred = True
        self._features = features
        self._columns_for_ids = columns_for_ids
        self._output_path_prefix = output_path_prefix
        self._output_format = output_format
        self._write_mem_limit = write_mem_limit
        self._write_labels = write_labels

        output_path, prefix = os.path.split(output_path_prefix)
        ref_filename = "ref"
        alt_filename = "alt"
        if len(prefix) > 0:
            ref_filename = "{0}.{1}".format(prefix, ref_filename)
            alt_filename = "{0}.{1}".format(prefix, alt_filename)
        ref_filepath = os.path.join(output_path, ref_filename)
        alt_filepath = os.path.join(output_path, alt_filename)

        self._ref_writer = WritePredictionsHandler(
            features,
            columns_for_ids,
            ref_filepath,
            output_format,
            output_size=output_size,
            write_mem_limit=write_mem_limit // 2,
            write_labels=write_labels)

        self._alt_writer = WritePredictionsHandler(
            features,
            columns_for_ids,
            alt_filepath,
            output_format,
            output_size=output_size,
            write_mem_limit=write_mem_limit // 2,
            write_labels=False)

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 base_predictions):
        self._ref_writer.handle_batch_predictions(
            base_predictions, batch_ids)
        self._alt_writer.handle_batch_predictions(
            batch_predictions, batch_ids)

    def write_to_file(self):
        self._ref_writer.write_to_file()
        self._alt_writer.write_to_file()
