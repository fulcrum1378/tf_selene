from abc import ABCMeta
from abc import abstractmethod
import os


class Sampler(metaclass=ABCMeta):
    BASE_MODES = ("train", "validate")

    def __init__(self, features, save_datasets=None, output_dir=None):
        if save_datasets is None:
            save_datasets = list()
        self.modes = list(self.BASE_MODES)
        self.mode = None
        self._features = features
        self._save_datasets = {}
        for mode in save_datasets:
            self._save_datasets[mode] = []
        self._output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    def set_mode(self, mode):
        if mode not in self.modes:
            raise ValueError(
                "Tried to set mode to be '{0}' but the only valid modes are "
                "{1}".format(mode, self.modes))
        self.mode = mode

    @abstractmethod
    def get_feature_from_index(self, index):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, batch_size=1, mode=None):
        raise NotImplementedError()

    @abstractmethod
    def get_data_and_targets(self, batch_size, n_samples, mode=None):
        raise NotImplementedError()

    @abstractmethod
    def get_validation_set(self, batch_size, n_samples=None):
        raise NotImplementedError()

    @abstractmethod
    def get_test_set(self, batch_size, n_samples=None):
        raise NotImplementedError()

    @abstractmethod
    def save_dataset_to_file(self, mode, close_filehandle=False):
        raise NotImplementedError()
