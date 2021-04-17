import logging
import os
from typing import Dict, List
import warnings

import numpy as np
import tensorflow as tf
from torch import load
import torch.nn as nn

from .samplers import Sampler
from .sequences import Genome
from .utils import _is_lua_trained_model
from .utils import initialize_logger
from .utils import load_model_from_state_dict
from .utils import PerformanceMetrics

logger = logging.getLogger("selene")


class EvaluateModel(object):
    def __init__(self,
                 model: nn.Module,
                 criterion,  # extends torch.nn._Loss
                 data_sampler: Sampler,  # must be a subclass of Sampler NOT itself
                 features: List[str],
                 trained_model_path: str,
                 output_dir: str,
                 batch_size: int = 64,
                 n_test_samples: int = None,
                 report_gt_feature_n_positives: int = 10,
                 use_cuda: bool = False,
                 # data_parallel: bool = False,
                 use_features_ord: List[str] = None):
        self.criterion = criterion
        trained_model = load(
            trained_model_path, map_location=lambda storage, location: storage)
        if "state_dict" in trained_model:
            self.model = load_model_from_state_dict(trained_model["state_dict"], model)
        else:
            self.model = load_model_from_state_dict(trained_model, model)
        self.model.eval()
        self.sampler = data_sampler
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.features = features
        self._use_ixs = list(range(len(features)))
        if use_features_ord is not None:
            feature_ixs = {f: ix for (ix, f) in enumerate(features)}
            self._use_ixs = []
            self.features = []
            for f in use_features_ord:
                if f in feature_ixs:
                    self._use_ixs.append(feature_ixs[f])
                    self.features.append(f)
                else:
                    warnings.warn(("Feature {0} in `use_features_ord` "
                                   "does not match any features in the list "
                                   "`features` and will be skipped.").format(f))
            self._write_features_ordered_to_file()

        initialize_logger(
            os.path.join(self.output_dir, "{0}.log".format(__name__)),
            verbosity=2)

        self.use_cuda = use_cuda
        if self.use_cuda: self.model.cuda()
        self.batch_size = batch_size
        self._metrics = PerformanceMetrics(
            self._get_feature_from_index,
            report_gt_feature_n_positives=report_gt_feature_n_positives)
        self._test_data, self._all_test_targets = \
            self.sampler.get_data_and_targets(self.batch_size, n_test_samples)
        self._all_test_targets = self._all_test_targets[:, self._use_ixs]

        if (hasattr(self.sampler, "reference_sequence") and
                isinstance(self.sampler.reference_sequence, Genome)):
            Genome.update_bases_order(['A', 'G', 'C', 'T'] if _is_lua_trained_model(model)
                                      else ['A', 'C', 'G', 'T'])

    def _write_features_ordered_to_file(self):
        fp = os.path.join(self.output_dir, 'use_features_ord.txt')
        with open(fp, 'w+') as file_handle:
            for f in self.features:
                file_handle.write('{0}\n'.format(f))

    def _get_feature_from_index(self, index):
        return self.features[index]

    def evaluate(self) -> Dict:
        batch_losses = []
        all_predictions = []
        for (inputs, targets) in self._test_data:
            inputs = tf.Tensor(inputs)
            targets = tf.Tensor(targets[:, self._use_ixs])

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = tf.Variable(inputs, trainable=False)
            targets = tf.Variable(targets, trainable=False)

            if _is_lua_trained_model(self.model):
                predictions = self.model.forward(
                    inputs.transpose(1, 2).contiguous().unsqueeze_(2))
            else:
                predictions = self.model.forward(inputs.transpose(1, 2))
            predictions = predictions[:, self._use_ixs]
            loss = self.criterion(predictions, targets)

            all_predictions.append(predictions.data.cpu().numpy())
            batch_losses.append(loss.item())
        all_predictions = np.vstack(all_predictions)

        average_scores = self._metrics.update(all_predictions, self._all_test_targets)

        self._metrics.visualize(all_predictions, self._all_test_targets, self.output_dir)
        np.savez_compressed(
            os.path.join(self.output_dir, "test_predictions.npz"),
            data=all_predictions)
        np.savez_compressed(
            os.path.join(self.output_dir, "test_targets.npz"),
            data=self._all_test_targets)
        loss = np.average(batch_losses)
        logger.info("test loss: {0}".format(loss))
        for name, score in average_scores.items():
            logger.info("test {0}: {1}".format(name, score))

        test_performance = os.path.join(
            self.output_dir, "test_performance.txt")
        feature_scores_dict = self._metrics.write_feature_scores_to_file(test_performance)

        return feature_scores_dict
