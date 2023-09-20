import yaml
from absl import flags
from absl import app
from asr import utils
import tensorflow as tf
from pathlib import Path
from asr.losses import ctc_loss
from asr.schedulers import warmup_cosine_scheduler
from asr.models import deep_speech_2, ctc_decoder
from asr.datasets import audio_dataset_dataloader
from asr.metrics.word_error_rate import WordErrorRate

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


def main(_):
    config_path = FLAGS.config
    output_dir = FLAGS.output_dir

    assert Path(config_path).is_file(
    ), f'Config file found error. No such file {FLAGS.config}.'
    if not Path(output_dir).is_dir():
        Path(output_dir).mkdir(parents=True)

    logger = utils.create_logger(output_dir, name='train')

    with open(config_path, 'r') as cfg:
        config = yaml.safe_load(cfg)

    utils.set_memory_growth()
    strategy = utils.get_strategy(logger=logger)

    logger.info(f'Num replicas in sync: {strategy.num_replicas_in_sync}')

    precision = utils.get_precision(config.get('precision', 'fp32'))
    policy = tf.keras.mixed_precision.Policy(precision)
    tf.keras.mixed_precision.set_global_policy(policy)

    logger.info(f'Precision: {policy.compute_dtype}')

    train_loader = audio_dataset_dataloader.AudioDatasetLoader(
        **(config['data']['params']))
    train_ds = train_loader.get_dataset()

    loss = ctc_loss.ctc_loss

    logger.info(f'Loss: {loss.__name__}')

    scheduler = warmup_cosine_scheduler.WarmupCosineScheduler(10, 1e-4, 10)

    logger.info(f'Scheduler params: {scheduler.get_config()}')

    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

    logger.info(f'Optimizer params: {optimizer.get_config()}')

    with strategy.scope():
        decoder = ctc_decoder.CTCDecoder(train_loader.num_to_char)
        metrics = [WordErrorRate(decoder)]
        model = deep_speech_2.DeepSpeech2(**(config['model']['params']))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(train_ds, epochs=10)
    decoder = ctc_decoder.CTCDecoder(train_loader.num_to_char)
    m = WordErrorRate(decoder)
    for x, y in train_ds.take(1):
        outputs = model(x)
        print(m(y, outputs))


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
