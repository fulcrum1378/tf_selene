from abc import ABCMeta
from abc import abstractmethod


class Target(metaclass=ABCMeta):
    @abstractmethod
    def get_feature_data(self, *args, **kwargs):
        raise NotImplementedError()
