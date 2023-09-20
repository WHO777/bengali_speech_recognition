import traceback
from typing import Dict, List

import tensorflow as tf

from asr.metrics.word_error_rate import WordErrorRate
from asr.models.ctc_decoder import CTCDecoder


def _parse_singe_config(
    config: Dict,
    num_to_char_lookup: tf.keras.layers.StringLookup = None
) -> tf.keras.metrics.Metric:
    metric_name = config['type']
    metric_params = config.get('params', {})
    decoder_cfg = metric_params.pop('decoder', None)
    if decoder_cfg is not None and num_to_char_lookup is None:
        decoder_name = decoder_cfg['type']
        decoder_params = decoder_cfg.get('params', {})
        try:
            decoder_type = eval(decoder_name)
        except NameError as err:
            traceback.print_tb(err.__traceback__)
            raise ValueError(f'Undefined decoder type {metric_name}.')
        decoder = decoder_type(num_to_char_lookup=num_to_char_lookup,
                               **decoder_params)
    else:
        decoder = None
    try:
        metric_type = eval(metric_name)
    except NameError as err:
        traceback.print_tb(err.__traceback__)
        raise ValueError(f'Undefined metric type {metric_name}.')
    metric = metric_type(decoder=decoder, **metric_params)
    return metric


def get_metrics(
    configs: Dict,
    num_to_char_lookup: tf.keras.layers.StringLookup = None
) -> List[tf.keras.metrics.Metric]:
    metrics = []
    for cfg in configs:
        metric = _parse_singe_config(cfg,
                                     num_to_char_lookup=num_to_char_lookup)
        metrics.append(metric)
    return metrics
