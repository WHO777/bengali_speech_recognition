import yaml
from absl import flags
from absl import app
from asr import utils
import tensorflow as tf
from pathlib import Path
from asr.losses import get_loss
from asr.schedulers import get_scheduler
from asr.optimizers import get_optimizer
from asr.models import get_model
from asr.datasets import get_dataloader
from asr.metrics import get_metrics

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string('config',
                        default=None,
                        help='training config (yaml file).',
                        required=True)
    flags.DEFINE_string('output_dir',
                        default=None,
                        help='dir where all training files will be saved.',
                        required=True)
    flags.DEFINE_string('checkpoint',
                        default=None,
                        help='saved model path')


def main(_):
    config_path = Path(FLAGS.config)
    output_dir = Path(FLAGS.output_dir)

    assert config_path.is_file(
    ), f'Config file found error. No such file {FLAGS.config}.'
    if not output_dir.is_dir():
        Path(output_dir).mkdir(parents=True)

    logger = utils.create_logger(output_dir, name='train')

    with open(config_path, 'r') as cfg:
        config = yaml.safe_load(cfg)

    utils.set_memory_growth()
    strategy = utils.get_strategy(logger=logger)

    logger.info(f'Num replicas in sync: {strategy.num_replicas_in_sync}')

    train_config = config['train']

    precision = utils.get_precision(train_config.get('precision', 'fp32'))
    policy = tf.keras.mixed_precision.Policy(precision)
    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info(f'Precision: {policy.compute_dtype}')

    train_dataloader = get_dataloader(config['train_dataloader'])
    train_ds = train_dataloader.get_dataset()
    if config.get('val_dataloader', None) is not None:
        val_dataloader = get_dataloader(config['val_dataloader'])
        val_ds = val_dataloader.get_dataset()
    else:
        val_ds = None

    loss = get_loss(config['loss'])
    logger.info(f'Loss: {loss.name if hasattr(loss, "name") else loss.__class__.__name__}')

    scheduler = get_scheduler(config['scheduler'])
    logger.info(f'Scheduler: {scheduler.name if hasattr(scheduler, "name") else scheduler.__class__.__name__}')

    optimizer = get_optimizer(config['optimizer'])
    logger.info(f'Optimizer params: {optimizer.name if hasattr(optimizer, "name") else optimizer.__class__.__name__}')

    with strategy.scope():
        metrics = get_metrics(config['metrics'], num_to_char_lookup=train_dataloader.num_to_char)
        model = get_model(config['model'])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if FLAGS.checkpoint is not None:
        model.load_checkpoint(FLAGS.checkpoint, logger=logger)

    model.fit(train_ds, epochs=train_config['epochs'], validation_data=val_ds)


if __name__ == '__main__':
    define_flags()
    app.run(main)
