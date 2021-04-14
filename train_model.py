import logging
import math
import os
import shutil
from time import strftime
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from .utils import initialize_logger
from .utils import load_model_from_state_dict
from .utils import PerformanceMetrics

logger = logging.getLogger("selene")


def _metrics_logger(name, out_filepath):
    logger = logging.getLogger("{0}".format(name))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handle = logging.FileHandler(
        os.path.join(out_filepath, "{0}.txt".format(name)))
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)
    return logger


class TrainModel(object):
    def __init__(self,
                 model,
                 data_sampler,
                 loss_criterion,
                 optimizer_class,
                 optimizer_kwargs,
                 batch_size,
                 max_steps,
                 report_stats_every_n_steps,
                 output_dir,
                 save_checkpoint_every_n_steps=1000,
                 save_new_checkpoints_after_n_steps=None,
                 report_gt_feature_n_positives=10,
                 n_validation_samples=None,
                 n_test_samples=None,
                 cpu_n_threads=1,
                 use_cuda=False,
                 data_parallel=False,
                 logging_verbosity=2,
                 checkpoint_resume=None,
                 metrics=dict(roc_auc=roc_auc_score,
                              average_precision=average_precision_score),
                 use_scheduler=True):
        """
        Constructs a new `TrainModel` object.
        """
        self.model = model
        self.sampler = data_sampler
        self.criterion = loss_criterion
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_kwargs)

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
                    "maximum number of steps: {2}".format(
            self.batch_size,
            self.nth_step_report_stats,
            self.max_steps))

        torch.set_num_threads(cpu_n_threads)

        self.use_cuda = use_cuda
        self.data_parallel = data_parallel

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            logger.debug("Wrapped model in DataParallel")

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
        self._metrics = metrics
        self._n_validation_samples = n_validation_samples
        self._n_test_samples = n_test_samples
        self._use_scheduler = use_scheduler

        self._init_train()
        self._init_validate()
        if "test" in self.sampler.modes:
            self._init_test()
        if checkpoint_resume is not None:
            self._load_checkpoint(checkpoint_resume)

    def _load_checkpoint(self, checkpoint_resume):
        checkpoint = torch.load(
            checkpoint_resume,
            map_location=lambda storage, location: storage)
        if "state_dict" not in checkpoint:
            raise ValueError(
                ("'state_dict' not found in file {0} "
                 "loaded with method `torch.load`. Selene does not support "
                 "continued training of models that were not originally "
                 "trained using Selene.").format(checkpoint_resume))

        self.model = load_model_from_state_dict(
            checkpoint["state_dict"], self.model)

        self._start_step = checkpoint["step"]
        if self._start_step >= self.max_steps:
            self.max_steps += self._start_step

        self._min_loss = checkpoint["min_loss"]
        self.optimizer.load_state_dict(
            checkpoint["optimizer"])
        if self.use_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        logger.info(
            ("Resuming from checkpoint: step {0}, min loss {1}").format(
                self._start_step, self._min_loss))

    def _init_train(self):
        self._start_step = 0
        self._train_logger = _metrics_logger(
            "{0}.train".format(__name__), self.output_dir)
        self._train_logger.info("loss")
        if self._use_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                'min',
                patience=16,
                verbose=True,
                factor=0.8)
        self._time_per_step = []
        self._train_loss = []

    def _init_validate(self):
        self._min_loss = float(
            "inf")  # TODO: Should this be set when it is used later? Would need to if we want to train model 2x in one run.
        self._create_validation_set(n_samples=self._n_validation_samples)
        self._validation_metrics = PerformanceMetrics(
            self.sampler.get_feature_from_index,
            report_gt_feature_n_positives=self._report_gt_feature_n_positives,
            metrics=self._metrics)
        self._validation_logger = _metrics_logger(
            "{0}.validation".format(__name__), self.output_dir)

        self._validation_logger.info("\t".join(["loss"] +
                                               sorted([x for x in self._validation_metrics.metrics.keys()])))

    def _init_test(self):
        self._test_data = None
        self._n_test_samples = self._n_test_samples
        self._test_metrics = PerformanceMetrics(
            self.sampler.get_feature_from_index,
            report_gt_feature_n_positives=self._report_gt_feature_n_positives,
            metrics=self._metrics)

    def _create_validation_set(self, n_samples=None):
        """
        Generates the set of validation examples.

        Parameters
        ----------
        n_samples : int or None, optional
            Default is `None`. The size of the validation set. If `None`,
            will use all validation examples in the sampler.

        """
        logger.info("Creating validation dataset.")
        t_i = time()
        self._validation_data, self._all_validation_targets = \
            self.sampler.get_validation_set(
                self.batch_size, n_samples=n_samples)
        t_f = time()
        logger.info(("{0} s to load {1} validation examples ({2} validation "
                     "batches) to evaluate after each training step.").format(
            t_f - t_i,
            len(self._validation_data) * self.batch_size,
            len(self._validation_data)))

    def create_test_set(self):
        """
        Loads the set of test samples.
        We do not create the test set in the `TrainModel` object until
        this method is called, so that we avoid having to load it into
        memory until the model has been trained and is ready to be
        evaluated.

        """
        logger.info("Creating test dataset.")
        t_i = time()
        self._test_data, self._all_test_targets = \
            self.sampler.get_test_set(
                self.batch_size, n_samples=self._n_test_samples)
        t_f = time()
        logger.info(("{0} s to load {1} test examples ({2} test batches) "
                     "to evaluate after all training steps.").format(
            t_f - t_i,
            len(self._test_data) * self.batch_size,
            len(self._test_data)))
        np.savez_compressed(
            os.path.join(self.output_dir, "test_targets.npz"),
            data=self._all_test_targets)

    def _get_batch(self):
        """
        Fetches a mini-batch of examples

        Returns
        -------
        tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the examples and targets.

        """
        t_i_sampling = time()
        batch_sequences, batch_targets = self.sampler.sample(
            batch_size=self.batch_size)
        t_f_sampling = time()
        logger.debug(
            ("[BATCH] Time to sample {0} examples: {1} s.").format(
                self.batch_size,
                t_f_sampling - t_i_sampling))
        return (batch_sequences, batch_targets)

    def _checkpoint(self):
        checkpoint_dict = {
            "step": self.step,
            "arch": self.model.__class__.__name__,
            "state_dict": self.model.state_dict(),
            "min_loss": self._min_loss,
            "optimizer": self.optimizer.state_dict()
        }
        if self._save_new_checkpoints is not None and \
                self._save_new_checkpoints >= self.step:
            checkpoint_filename = "checkpoint-{0}".format(
                strftime("%m%d%H%M%S"))
            self._save_checkpoint(
                checkpoint_dict, False, filename=checkpoint_filename)
            logger.debug("Saving checkpoint `{0}.pth.tar`".format(
                checkpoint_filename))
        else:
            self._save_checkpoint(
                checkpoint_dict, False)

    def train_and_validate(self):
        """
        Trains the model and measures validation performance.

        """
        for step in range(self._start_step, self.max_steps):
            self.step = step
            self.train()

            if step % self.nth_step_save_checkpoint == 0:
                self._checkpoint()
            if self.step and self.step % self.nth_step_report_stats == 0:
                self.validate()

        self.sampler.save_dataset_to_file("train", close_filehandle=True)

    def train(self):
        """
        Trains the model on a batch of data.

        Returns
        -------
        float
            The training loss.

        """
        t_i = time()
        self.model.train()
        self.sampler.set_mode("train")

        inputs, targets = self._get_batch()
        inputs = torch.Tensor(inputs)
        targets = torch.Tensor(targets)

        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs)
        targets = Variable(targets)

        predictions = self.model(inputs.transpose(1, 2))
        loss = self.criterion(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._train_loss.append(loss.item())
        t_f = time()

        self._time_per_step.append(t_f - t_i)
        if self.step and self.step % self.nth_step_report_stats == 0:
            logger.info(("[STEP {0}] average number "
                         "of steps per second: {1:.1f}").format(
                self.step, 1. / np.average(self._time_per_step)))
            self._train_logger.info(np.average(self._train_loss))
            logger.info("training loss: {0}".format(np.average(self._train_loss)))
            self._time_per_step = []
            self._train_loss = []

    def _evaluate_on_data(self, data_in_batches):
        """
        Makes predictions for some labeled input data.

        Parameters
        ----------
        data_in_batches : list(tuple(numpy.ndarray, numpy.ndarray))
            A list of tuples of the data, where the first element is
            the example, and the second element is the label.

        Returns
        -------
        tuple(float, list(numpy.ndarray))
            Returns the average loss, and the list of all predictions.

        """
        self.model.eval()

        batch_losses = []
        all_predictions = []

        for (inputs, targets) in data_in_batches:
            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets)

            if self.use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)

                predictions = self.model(inputs.transpose(1, 2))
                loss = self.criterion(predictions, targets)

                all_predictions.append(
                    predictions.data.cpu().numpy())

                batch_losses.append(loss.item())
        all_predictions = np.vstack(all_predictions)
        return np.average(batch_losses), all_predictions

    def validate(self):
        """
        Measures model validation performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the validation set.

        """
        validation_loss, all_predictions = self._evaluate_on_data(
            self._validation_data)
        valid_scores = self._validation_metrics.update(
            all_predictions, self._all_validation_targets)
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

        # scheduler update
        if self._use_scheduler:
            self.scheduler.step(
                math.ceil(validation_loss * 1000.0) / 1000.0)

        # save best_model
        if validation_loss < self._min_loss:
            self._min_loss = validation_loss
            self._save_checkpoint({
                "step": self.step,
                "arch": self.model.__class__.__name__,
                "state_dict": self.model.state_dict(),
                "min_loss": self._min_loss,
                "optimizer": self.optimizer.state_dict()}, True)
            logger.debug("Updating `best_model.pth.tar`")
        logger.info("validation loss: {0}".format(validation_loss))

    def evaluate(self):
        """
        Measures the model test performance.

        Returns
        -------
        dict
            A dictionary, where keys are the names of the loss metrics,
            and the values are the average value for that metric over
            the test set.

        """
        if self._test_data is None:
            self.create_test_set()
        average_loss, all_predictions = self._evaluate_on_data(
            self._test_data)

        average_scores = self._test_metrics.update(all_predictions,
                                                   self._all_test_targets)
        np.savez_compressed(
            os.path.join(self.output_dir, "test_predictions.npz"),
            data=all_predictions)

        for name, score in average_scores.items():
            logger.info("test {0}: {1}".format(name, score))

        test_performance = os.path.join(
            self.output_dir, "test_performance.txt")
        feature_scores_dict = self._test_metrics.write_feature_scores_to_file(
            test_performance)

        average_scores["loss"] = average_loss

        self._test_metrics.visualize(
            all_predictions, self._all_test_targets, self.output_dir)

        return (average_scores, feature_scores_dict)

    def _save_checkpoint(self,
                         state,
                         is_best,
                         filename="checkpoint"):
        logger.debug("[TRAIN] {0}: Saving model state to file.".format(
            state["step"]))
        cp_filepath = os.path.join(
            self.output_dir, filename)
        torch.save(state, "{0}.pth.tar".format(cp_filepath))
        if is_best:
            best_filepath = os.path.join(self.output_dir, "best_model")
            shutil.copyfile("{0}.pth.tar".format(cp_filepath),
                            "{0}.pth.tar".format(best_filepath))
