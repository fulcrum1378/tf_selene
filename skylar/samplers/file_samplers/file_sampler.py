from abc import ABCMeta
from abc import abstractmethod


class FileSampler(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, batch_size=1):
        raise NotImplementedError()

    @abstractmethod
    def get_data_and_targets(self, batch_size, n_samples):
        raise NotImplementedError()

    @abstractmethod
    def get_data(self, batch_size, n_samples):
        raise NotImplementedError()
