from .handler import PredictionsHandler


class DiffScoreHandler(PredictionsHandler):
    def __init__(self,
                 features,
                 columns_for_ids,
                 output_path_prefix,
                 output_format,
                 output_size=None,
                 write_mem_limit=1500,
                 write_labels=True):
        super(DiffScoreHandler, self).__init__(
            features,
            columns_for_ids,
            output_path_prefix,
            output_format,
            output_size=output_size,
            write_mem_limit=write_mem_limit,
            write_labels=write_labels)

        self.needs_base_pred = True
        self._results = []
        self._samples = []

        self._features = features
        self._columns_for_ids = columns_for_ids
        self._output_path_prefix = output_path_prefix
        self._output_format = output_format
        self._write_mem_limit = write_mem_limit
        self._write_labels = write_labels

        self._create_write_handler("diffs")

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 baseline_predictions):
        diffs = batch_predictions - baseline_predictions
        self._results.append(diffs)
        self._samples.append(batch_ids)
        if self._reached_mem_limit():
            self.write_to_file()

    def write_to_file(self):
        super().write_to_file()
