from abc import ABCMeta, abstractmethod
from typing import Optional

import tensorflow as tf


class BaseDataLoader(metaclass=ABCMeta):

    @abstractmethod
    def get_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError()


class TextDataLoader(BaseDataLoader, metaclass=ABCMeta):

    def __init__(self):
        self.num_to_char: Optional[tf.keras.layers.StringLookup] = None
        self.char_to_num: Optional[tf.keras.layers.StringLookup] = None

    @abstractmethod
    def get_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError()
