import traceback
from typing import Dict

import tensorflow as tf


def get_optimizer(config: Dict) -> tf.keras.optimizers.Optimizer:
    optimizer_name = config['type']
    optimizer_params = config.get('params', {})
    try:
        optimizer_type = getattr(tf.keras.optimizers, optimizer_name)
    except AttributeError as err:
        traceback.print_tb(err.__traceback__)
        raise ValueError(f'Undefined optimizer type {optimizer_name}.')
    optimizer = optimizer_type(**optimizer_params)
    return optimizer
