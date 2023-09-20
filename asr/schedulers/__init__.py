import traceback
from typing import Dict

import tensorflow as tf

from asr.schedulers.warmup_cosine_scheduler import WarmupCosineScheduler


def get_scheduler(
        config: Dict) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    scheduler_name = config['type']
    scheduler_params = config.get('params', {})
    try:
        scheduler_type = eval(scheduler_name)
    except NameError as err:
        traceback.print_tb(err.__traceback__)
        raise ValueError(f'Undefined scheduler type {scheduler_name}.')
    scheduler = scheduler_type(**scheduler_params)
    return scheduler
