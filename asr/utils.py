import functools
import logging
import platform
import sys
import traceback
from pathlib import Path
from typing import NoReturn

import tensorflow as tf


def set_memory_growth(devices=None, device_type='GPU') -> NoReturn:
    if devices is None:
        devices = tf.config.list_physical_devices(device_type)
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)


def get_strategy(logger: logging.Logger = None) -> tf.distribute.Strategy:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if platform.system() == 'Windows':
            cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
        else:
            cross_device_ops = None
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=cross_device_ops)
    else:
        strategy = tf.distribute.OneDeviceStrategy('CPU')
    if logger is not None:
        if gpus:
            gpu_idxs = [f"{gpu.name.split(':')[-1]}" for gpu in gpus]
            gpu_names = [
                tf.config.experimental.get_device_details(gpu)['device_name']
                for gpu in gpus
            ]
            gpus_log_info = 'GPUs found: ' + ''.join([
                f'GPU:{gpu_idx}, {gpu_name}'
                for (gpu_idx, gpu_name) in zip(gpu_idxs, gpu_names)
            ])
            logger.info(gpus_log_info)
    return strategy


def get_precision(name='fp32'):
    precisions = {
        'fp32': 'float32',
        'fp16': 'mixed_float16',
    }
    precision = precisions.get(name, None)
    if precision is None:
        raise RuntimeError(
            f'Unknown precision name {name}. Supported precisions: {list(precisions.keys())}'
        )
    return precision


@functools.lru_cache()
def create_logger(output_dir=None, name=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    if output_dir is not None:
        file_handler = logging.FileHandler(Path(output_dir) / 'log.txt')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger


def load_checkpoint(model,
                    checkpoint_path,
                    skip_mismatch=False,
                    by_name=False,
                    options=None,
                    logger=None):
    if Path(checkpoint_path).is_dir():
        checkpoint_path = str(
            Path(checkpoint_path) / 'variables' / 'variables')
    try:
        model.load_weights(checkpoint_path,
                           skip_mismatch=skip_mismatch,
                           by_name=by_name,
                           options=options)
        if logger is not None:
            logger.info(
                f'Successfully restored weights from {checkpoint_path} checkpoint.'
            )
    except Exception as err:
        traceback.print_tb(err.__traceback__)
        if logger is not None:
            logger.warning(
                'Weights cannot be restored completely from checkpoint {}, an error occurred. '
                'Trying to restore all weights except the weights of the last layer.'
            )
        no_top_model = tf.keras.Model(model.layers[0].input,
                                      model.layers[-2].output)
        try:
            for i in range(len(no_top_model.layers)):
                model.layers[i].set_weights(
                    no_top_model.layers[i].get_weights())
            if logger is not None:
                logger.warning(
                    f'All weights except the weights of the last layers were restored. '
                    f'The weights of the last layer cannot be restored from the checkpoint {checkpoint_path}.'
                )
        except Exception as err:
            traceback.print_tb(err.__traceback__)
            if logger is not None:
                logger.error(
                    f'Weights cannot be restored from the checkpoint {checkpoint_path}.'
                )
