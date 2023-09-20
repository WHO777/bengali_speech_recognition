import traceback
from typing import Dict

import tensorflow as tf

from asr.models.deep_speech_2 import DeepSpeech2


def get_model(config: Dict) -> tf.keras.Model:
    model_name = config['type']
    model_params = config.get('params', {})
    try:
        model_type = eval(model_name)
    except NameError as err:
        traceback.print_tb(err.__traceback__)
        raise ValueError(f'Undefined model type {model_name}.')
    model = model_type(**model_params)
    return model
