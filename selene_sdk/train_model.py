import logging
import os
import shutil
from time import strftime
from time import time
from typing import Dict, Tuple, Type

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from .samplers import Sampler
from .utils import initialize_logger
from .utils import load_model_from_state_dict
from .utils import PerformanceMetrics

logger = logging.getLogger("selene")


def _metrics_logger(name: str, out_filepath: str) -> logging:
    my_logger = logging.getLogger("{0}".format(name))
    my_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handle = logging.FileHandler(os.path.join(out_filepath, "{0}.txt".format(name)))
    file_handle.setFormatter(formatter)
    my_logger.addHandler(file_handle)
    return my_logger


class TrainModel(object):
    def __init__(self,
                 model: Type[tf.Module],
                 data_sampler: Sampler,
                 loss_criterion: tf.keras.losses.Loss,
                 optimizer_class: Type[tf.keras.optimizers.Optimizer],  # currently tfa.optimizers.SGDW
                 optimizer_kwargs: Dict,
                 batch_size: int,
                 max_steps: int,
                 report_stats_every_n_steps: int,
                 output_dir: str,
                 save_checkpoint_every_n_steps: int = 1000,
                 save_new_checkpoints_after_n_steps: int = None,
                 report_gt_feature_n_positives: int = 10,
                 n_validation_samples: int = None,
                 n_test_samples: int = None,
                 cpu_n_threads: int = 1,
                 use_cuda: bool = False,
                 # data_parallel: bool = False,
                 logging_verbosity: int = 2,
                 checkpoint_resume: str = None):
        self.model = model
        self.sampler = data_sampler
        self.criterion = loss_criterion
        self.optimizer = optimizer_class(self.model.variables, **optimizer_kwargs)

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.nth_step_report_stats = report_stats_every_n_steps
        self.nth_step_save_checkpoint = None

        if not save_checkpoint_every_n_steps:
            self.nth_step_save_checkpoint = report_stats_every_n_steps
        else:
            self.nth_step_save_checkpoint = save_checkpoint_every_n_steps

        self._save_new_checkpoints = save_new_checkpoints_after_n_steps

        logger.info("Training parameters set: batch size {0}, "
                    "number of steps per 'epoch': {1}, "
                    "maximum number of steps: {2}".format(self.batch_size,
                                                          self.nth_step_report_stats,
                                                          self.max_steps))

        tf.config.threading.set_intra_op_parallelism_threads(cpu_n_threads)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            logger.debug("Set modules to use CUDA")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        initialize_logger(
            os.path.join(self.output_dir, "{0}.log".format(__name__)),
            verbosity=logging_verbosity)

        self._report_gt_feature_n_positives = report_gt_feature_n_positives
        self._metrics = dict(roc_auc=roc_auc_score, average_precision=average_precision_score)
        self._n_validation_samples = n_validation_samples
        self._n_test_samples = n_test_samples

        self._init_train()
        self._init_validate()
        if "test" in self.sampler.modes:
            self._init_test()
        if checkpoint_resume is not None:
            self._load_checkpoint(checkpoint_resume)

        self._test_data = self._all_test_targets = self.step = self._time_per_step = \
            self._train_loss = self._min_loss = None

    def _load_checkpoint(self, checkpoint_resume: str) -> None:
        checkpoint = tf.saved_model.load(checkpoint_resume, map_location=lambda storage, location: storage)
        if "state_dict" not in checkpoint:
            raise ValueError("'state_dict' not found in file {0} ".format(checkpoint_resume))

        self.model = load_model_from_state_dict(checkpoint["state_dict"], self.model)

        self._start_step = checkpoint["step"]
        if self._start_step >= self.max_steps:
            self.max_steps += self._start_step

        self._min_loss = checkpoint["min_loss"]
        self.optimizer.load_state_dict(
            checkpoint["optimizer"])
        if self.use_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, tf.Tensor):
                        state[k] = v.cuda()

        logger.info("Resuming from checkpoint: step {0}, min loss {1}".format(self._start_step, self._min_loss))

    def _init_train(self) -> None:
        self._start_step = 0
        self._train_logger = _metrics_logger("{0}.train".format(__name__), self.output_dir)
        self._train_logger.info("loss")
        self._time_per_step = []
        self._train_loss = []

    def _init_validate(self) -> None:
        self._min_loss = float("inf")
        self._create_validation_set(n_samples=self._n_validation_samples)
        self._validation_metrics = PerformanceMetrics(
            self.sampler.get_feature_from_index,
            report_gt_feature_n_positives=self._report_gt_feature_n_positives,
            metrics=self._metrics)
        self._validation_logger = _metrics_logger("{0}.validation".format(__name__), self.output_dir)

        self._validation_logger.info(
            "\t".join(["loss"] + sorted([x for x in self._validation_metrics.metrics.keys()])))

    def _init_test(self) -> None:
        self._n_test_samples = self._n_test_samples
        self._test_metrics = PerformanceMetrics(
            self.sampler.get_feature_from_index,
            report_gt_feature_n_positives=self._report_gt_feature_n_positives,
            metrics=self._metrics)

    def _create_validation_set(self, n_samples: int = None) -> None:
        logger.info("Creating validation dataset.")
        t_i = time()
        self._validation_data, self._all_validation_targets = \
            self.sampler.get_validation_set(self.batch_size, n_samples=n_samples)
        t_f = time()
        logger.info(("{0} s to load {1} validation examples ({2} validation "
                     "batches) to evaluate after each training step.").format(
            t_f - t_i,
            len(self._validation_data) * self.batch_size,
            len(self._validation_data)))

    def create_test_set(self) -> None:
        logger.info("Creating test dataset.")
        t_i = time()
        self._test_data, self._all_test_targets = \
            self.sampler.get_test_set(self.batch_size, n_samples=self._n_test_samples)
        t_f = time()
        logger.info(("{0} s to load {1} test examples ({2} test batches) "
                     "to evaluate after all training steps.").format(
            t_f - t_i,
            len(self._test_data) * self.batch_size,
            len(self._test_data)))
        np.savez_compressed(
            os.path.join(self.output_dir, "test_targets.npz"),
            data=self._all_test_targets)

    def _get_batch(self) -> Tuple:
        t_i_sampling = time()
        batch_sequences, batch_targets = self.sampler.sample(
            batch_size=self.batch_size)
        t_f_sampling = time()
        logger.debug("[BATCH] Time to sample {0} examples: {1} s.".format(
            self.batch_size, t_f_sampling - t_i_sampling))
        return batch_sequences, batch_targets

    def _checkpoint(self) -> None:
        checkpoint_dict = {
            "step": self.step,
            "arch": self.model.__class__.__name__,
            "state_dict": self.model.state_dict(),
            "min_loss": self._min_loss,
            "optimizer": self.optimizer.state_dict()
        }
        if self._save_new_checkpoints is not None and self._save_new_checkpoints >= self.step:
            checkpoint_filename = "checkpoint-{0}".format(strftime("%m%d%H%M%S"))
            self._save_checkpoint(checkpoint_dict, False, filename=checkpoint_filename)
            logger.debug("Saving checkpoint `{0}.pb.tar`".format(checkpoint_filename))
        else:
            self._save_checkpoint(checkpoint_dict, False)

    def train_and_validate(self) -> None:
        for step in range(self._start_step, self.max_steps):
            self.step = step
            self.train()
            if step % self.nth_step_save_checkpoint == 0:
                self._checkpoint()
            if self.step and self.step % self.nth_step_report_stats == 0:
                self.validate()
        self.sampler.save_dataset_to_file("train", close_filehandle=True)

    def train(self) -> None:
        t_i = time()
        self.model.train()
        self.sampler.set_mode("train")

        inputs, targets = self._get_batch()
        inputs = tf.Tensor(inputs)
        targets = tf.Tensor(targets)

        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = tf.Variable(inputs)
        targets = tf.Variable(targets)

        predictions = self.model(inputs.transpose(1, 2))
        loss = self.criterion(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._train_loss.append(loss.item())
        t_f = time()

        self._time_per_step.append(t_f - t_i)
        if self.step and self.step % self.nth_step_report_stats == 0:
            logger.info("[STEP {0}] average number of steps per second: {1:.1f}".format(
                self.step, 1. / np.average(self._time_per_step)))
            self._train_logger.info(np.average(self._train_loss))
            logger.info("training loss: {0}".format(np.average(self._train_loss)))
            self._time_per_step = []
            self._train_loss = []

    def _evaluate_on_data(self, data_in_batches: Dict) -> Tuple:
        self.model.eval()

        batch_losses = list()
        all_predictions = list()

        for inputs, targets in data_in_batches:
            inputs = tf.Tensor(inputs)
            targets = tf.Tensor(targets)

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = tf.Variable(inputs, trainable=False)
            targets = tf.Variable(targets, trainable=False)

            predictions = self.model(inputs.transpose(1, 2))
            loss = self.criterion(predictions, targets)

            all_predictions.append(predictions.data.cpu().numpy())

            batch_losses.append(loss.item())
        all_predictions = np.vstack(all_predictions)
        return np.average(batch_losses), all_predictions

    def validate(self) -> None:
        validation_loss, all_predictions = self._evaluate_on_data(self._validation_data)
        valid_scores = self._validation_metrics.update(all_predictions, self._all_validation_targets)
        for name, score in valid_scores.items():
            logger.info("validation {0}: {1}".format(name, score))

        valid_scores["loss"] = validation_loss

        to_log = [str(validation_loss)]
        for k in sorted(self._validation_metrics.metrics.keys()):
            if k in valid_scores and valid_scores[k]:
                to_log.append(str(valid_scores[k]))
            else:
                to_log.append("NA")
        self._validation_logger.info("\t".join(to_log))

        # save best_model
        if validation_loss < self._min_loss:
            self._min_loss = validation_loss
            self._save_checkpoint({
                "step": self.step,
                "arch": self.model.__class__.__name__,
                "state_dict": self.model.state_dict(),
                "min_loss": self._min_loss,
                "optimizer": self.optimizer.state_dict()}, True)
            logger.debug("Updating `best_model.pb.tar`")
        logger.info("validation loss: {0}".format(validation_loss))

    def evaluate(self) -> Tuple:
        if self._test_data is None: self.create_test_set()
        average_loss, all_predictions = self._evaluate_on_data(self._test_data)
        average_scores = self._test_metrics.update(all_predictions, self._all_test_targets)
        np.savez_compressed(
            os.path.join(self.output_dir, "test_predictions.npz"),
            data=all_predictions)
        for name, score in average_scores.items():
            logger.info("test {0}: {1}".format(name, score))
        test_performance = os.path.join(self.output_dir, "test_performance.txt")
        feature_scores_dict = self._test_metrics.write_feature_scores_to_file(test_performance)
        average_scores["loss"] = average_loss
        self._test_metrics.visualize(all_predictions, self._all_test_targets, self.output_dir)
        return average_scores, feature_scores_dict

    def _save_checkpoint(self, state: Dict, is_best: bool, filename: str = "checkpoint") -> None:
        logger.debug("[TRAIN] {0}: Saving model state to file.".format(state["step"]))
        cp_filepath = os.path.join(self.output_dir, filename)
        tf.saved_model.save(state, "{0}.pb.tar".format(cp_filepath))
        if is_best:
            best_filepath = os.path.join(self.output_dir, "best_model")
            shutil.copyfile("{0}.pb.tar".format(cp_filepath), "{0}.pb.tar".format(best_filepath))
