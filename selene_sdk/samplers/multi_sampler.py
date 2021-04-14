import numpy as np
from torch.utils.data import DataLoader

from .sampler import Sampler


def MultiFileSampler(*args, **kwargs):
    from warnings import warn
    warn("MultiFileSampler is deprecated and will be removed from future "
         "versions of Selene. Please use MultiSampler instead.")
    return MultiSampler(*args, **kwargs)


class MultiSampler(Sampler):
    def __init__(self,
                 train_sampler,
                 validate_sampler,
                 features,
                 test_sampler=None,
                 mode="train",
                 save_datasets=[],
                 output_dir=None):
        super(MultiSampler, self).__init__(
            features,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self._samplers = {
            "train": train_sampler if isinstance(train_sampler, Sampler)
                else None,
            "validate": validate_sampler if isinstance(validate_sampler, Sampler)
                else None
        }

        self._dataloaders = {
            "train": train_sampler if isinstance(train_sampler, DataLoader)
                else None,
            "validate": validate_sampler if isinstance(validate_sampler, DataLoader)
                else None
        }

        self._iterators = {
            "train": iter(self._dataloaders["train"])
                if self._dataloaders["train"] else None,
            "validate": iter(self._dataloaders["validate"])
                if self._dataloaders["validate"] else None
        }

        self._index_to_feature = {i: f for (i, f) in enumerate(features)}

        if test_sampler is not None:
            self.modes.append("test")
            self._samplers["test"] = \
                test_sampler if isinstance(test_sampler, Sampler) else None
            self._dataloaders["test"] = \
                test_sampler if isinstance(test_sampler, DataLoader) else None

        self.mode = mode

    def set_mode(self, mode):
        if mode not in self.modes:
            raise ValueError(
                "Tried to set mode to be '{0}' but the only valid modes are "
                "{1}".format(mode, self.modes))
        self.mode = mode

    def _set_batch_size(self, batch_size, mode=None):
        if mode is None:
            mode = self.mode

        if self._dataloaders[mode]:
            batch_size_matched = True
            if self._dataloaders[mode].batch_sampler:
                if self._dataloaders[mode].batch_sampler.batch_size != batch_size:
                    self._dataloaders[mode].batch_sampler.batch_size = batch_size
                    batch_size_matched = False
            else:
                if self._dataloaders[mode].batch_size != batch_size:
                    self._dataloaders[mode].batch_size = batch_size
                    batch_size_matched = False

            if not batch_size_matched:
                print("Reset data loader for mode {0} to use the new batch "
                      "size: {1}.".format(mode, batch_size))
                self._iterators[mode] = iter(self._dataloaders[mode])

    def get_feature_from_index(self, index):
        return self._index_to_feature[index]

    def sample(self, batch_size=1, mode=None):
        mode = mode if mode else self.mode
        if self._samplers[mode]:
            return self._samplers[mode].sample(batch_size)
        else:
            self._set_batch_size(batch_size, mode=mode)
            try:
                data, targets = next(self._iterators[mode])
                return data.numpy(), targets.numpy()
            except StopIteration:
                # If DataLoader iterator reaches its length, reinitialize
                self._iterators[mode] = iter(self._dataloaders[mode])
                data, targets = next(self._iterators[mode])
                return data.numpy(), targets.numpy()

    def get_data_and_targets(self, batch_size, n_samples=None, mode=None):
        mode = mode if mode else self.mode
        if self._samplers[mode]:
            return self._samplers[mode].get_data_and_targets(
                batch_size, n_samples)
        else:
            if n_samples is None:
                if mode == 'validate':
                    n_samples = 32000
                elif mode == 'test':
                    n_samples = 640000
            self._set_batch_size(batch_size, mode=mode)
            data_and_targets = []
            targets_mat = []
            count = batch_size
            while count < n_samples:
                data, tgts = self.sample(batch_size=batch_size, mode=mode)
                data_and_targets.append((data, tgts))
                targets_mat.append(tgts)
                count += batch_size
            remainder = batch_size - (count - n_samples)
            data, tgts = self.sample(batch_size=remainder)
            data_and_targets.append((data, tgts))
            targets_mat.append(tgts)
            targets_mat = np.vstack(targets_mat)
            return data_and_targets, targets_mat

    def get_validation_set(self, batch_size, n_samples=None):
        return self.get_data_and_targets(
            batch_size, n_samples, mode="validate")

    def get_test_set(self, batch_size, n_samples=None):
        return self.get_data_and_targets(
            batch_size, n_samples, mode="test")

    def save_dataset_to_file(self, mode, close_filehandle=False):
        return None
