from abc import ABCMeta, abstractmethod

import tensorflow as tf


class BaseDataLoader(metaclass=ABCMeta):

    @abstractmethod
    def get_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError()
