from scipy.special import logit

from .handler import PredictionsHandler


class LogitScoreHandler(PredictionsHandler):
    def __init__(self,
                 features,
                 columns_for_ids,
                 output_path_prefix,
                 output_format,
                 output_size=None,
                 write_mem_limit=1500,
                 write_labels=True):
        super(LogitScoreHandler, self).__init__(
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

        self._create_write_handler("logits")

    def handle_batch_predictions(self,
                                 batch_predictions,
                                 batch_ids,
                                 baseline_predictions):
        baseline_predictions[baseline_predictions == 0] = 1e-24
        baseline_predictions[baseline_predictions >= 1] = 0.999999

        batch_predictions[batch_predictions == 0] = 1e-24
        batch_predictions[batch_predictions >= 1] = 0.999999

        logits = logit(batch_predictions) - logit(baseline_predictions)
        self._results.append(logits)
        self._samples.append(batch_ids)
        if self._reached_mem_limit():
            self.write_to_file()

    def write_to_file(self):
        super().write_to_file()
