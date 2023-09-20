import math

import tensorflow as tf


class WarmupCosineScheduler(tf.optimizers.schedules.LearningRateSchedule):

    def __init__(self,
                 num_warmup_steps,
                 lr_max,
                 num_training_steps,
                 warmup_method='log',
                 num_cycles=0.5):
        super().__init__()
        self.num_warmup_steps = num_warmup_steps
        self.lr_max = lr_max
        self.num_training_steps = num_training_steps
        self.warmup_method = warmup_method
        self.num_cycles = num_cycles

    def __call__(self, step):
        log_warmup = self.lr_max * 0.10**tf.cast(self.num_warmup_steps - step,
                                                 tf.float32)
        pol_warmup = self.lr_max * 2**-tf.cast(self.num_warmup_steps - step,
                                               tf.float32)
        warmup = tf.where(self.warmup_method == 'log', log_warmup, pol_warmup)
        progress = tf.cast(step - self.num_warmup_steps, tf.float32) / tf.cast(
            max(1, self.num_training_steps - self.num_warmup_steps),
            tf.float32)
        cos_lr = tf.maximum(
            0.0, 0.5 *
            (1.0 + tf.cos(math.pi * tf.cast(self.num_cycles, tf.float32) *
                          2.0 * progress))) * self.lr_max
        return tf.where(step < self.num_warmup_steps, warmup, cos_lr)

    def get_config(self):
        return {
            'num_warmup_steps': self.num_warmup_steps,
            'lr_max': self.lr_max,
            'num_training_steps': self.num_training_steps,
            'warmup_method': self.warmup_method,
            'num_cycles': self.num_cycles
        }
