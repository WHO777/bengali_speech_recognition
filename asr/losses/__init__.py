import traceback
from typing import Callable, Dict

import tensorflow as tf

from asr.losses.ctc_loss import CTCLoss


def get_loss(config: Dict) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    loss_name = config['type']
    loss_params = config.get('params', {})
    try:
        loss_type = eval(loss_name)
    except NameError as err:
        traceback.print_tb(err.__traceback__)
        raise ValueError(f'Undefined loss type {loss_name}.')
    loss = loss_type(**loss_params)
    return loss
