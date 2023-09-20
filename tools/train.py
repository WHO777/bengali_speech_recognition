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
                        help='save model path')


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
        metrics = get_metrics(config['metrics'])
        model = get_model(config['model'])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # model.fit(train_ds, epochs=train_config['epochs'], validation_data=val_ds)

    # fig = plt.figure(figsize=(8, 5))
    # for batch in train_ds.take(1):
    #     spectrogram = batch[0][0].numpy()
    #     print(spectrogram)
    #     spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    #     label = batch[1][0]
    #     # Spectrogram
    #     label = 'kekw'
    #     ax = plt.subplot(2, 1, 1)
    #     ax.imshow(spectrogram, vmax=1)
    #     ax.set_title(label)
    #     ax.axis("off")
    #     # Wav
    #     file = tf.io.read_file('/app/datasets/audio_dataset/audio_files/000073e5-e343-4b0f-a3c0-2828de3dbe37.wav')
    #     audio, _ = tf.audio.decode_wav(file)
    #     audio = audio.numpy()
    #     ax = plt.subplot(2, 1, 2)
    #     plt.plot(audio)
    #     ax.set_title("Signal Wave")
    #     ax.set_xlim(0, len(audio))
    #     fig.savefig('kekw.png')

    # row, col = 3, 3
    # plt.figure(figsize=(col * 5, row * 3))
    # imgs, _ = next(train_ds.take(1).as_numpy_iterator())
    # for idx in range(row * col):
    #     ax = plt.subplot(row, col, idx + 1)
    #     print(imgs.shape)
    #     lid.specshow(imgs[idx],
    #                  sr=16000,
    #                  hop_length=921,
    #                  fmin=0,
    #                  fmax=8000,
    #                  x_axis='time',
    #                  y_axis='mel',
    #                  cmap='coolwarm')
    # plt.tight_layout()
    # plt.show()
    # plt.savefig('kekw.png')


if __name__ == '__main__':
    from asr.models import ctc_decoder
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa.display as lid
    define_flags()
    app.run(main)
