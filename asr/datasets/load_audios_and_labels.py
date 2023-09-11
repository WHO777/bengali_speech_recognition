import os.path
import typing
from pathlib import Path
from typing import Sequence

import tensorflow as tf

from asr.datasets import base_dataloader, common


class LoadAudiosAndLabels(base_dataloader.BaseDataLoader):
    MAX_AUDIO_LEN: int = 200
    MAX_LABEL_LEN: int = 10

    SOS_TOKEN: str = '<sos>'
    EOS_TOKEN: str = '<eos>'
    PAD_TOKEN: str = '<pad>'
    SPACE_TOKEN: str = '|'
    OOV_TOKEN: str = '*'

    def __init__(self,
                 audio_paths: Sequence[str],
                 labels: Sequence[str] = None,
                 pad_mode: str = 'constant',
                 take_first: bool = True,
                 win_length: int = None,
                 hop_length: int = 513,
                 n_fft: int = 2048,
                 language: common.Language = common.Language.RUSSIAN,
                 max_label_len: int = MAX_LABEL_LEN,
                 sos_token: str = SOS_TOKEN,
                 eos_token: str = EOS_TOKEN,
                 pad_token: str = PAD_TOKEN,
                 space_token: str = SPACE_TOKEN,
                 OOV_TOKEN: str = OOV_TOKEN):
        self.audio_paths = audio_paths
        if labels is not None:
            vocabulary = common.LANGUAGE_TO_VOCABULARY_MAP.get(language, None)
            if vocabulary is None:
                raise ValueError(
                    f'Language {language} is not supported. Supported languages: {[l.name for l in common.LANGUAGE_TO_VOCABULARY_MAP]}'
                )
            vocabulary += [sos_token, eos_token, pad_token, space_token]
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=vocabulary, oov_token=OOV_TOKEN)
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(),
                oov_token=OOV_TOKEN,
                invert=True)
        self.labels = labels
        self.pad_mode = pad_mode
        self.take_first = take_first
        self.win_length = win_length or n_fft
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.max_label_len = max_label_len
        self.eos_token = eos_token
        self.sos_token = sos_token
        self.pad_token = pad_token
        self.space_token = space_token

    def decode_audio(self, audio_path: str) -> tf.Tensor:
        audio_bytes = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio_bytes)
        audio = tf.cast(audio, tf.float32)
        # stereo to mono
        if tf.shape(audio)[1] > 1:
            audio = audio[:, 0]
        return audio

    def crop_or_pad_sequence(self,
                             sequence: tf.Tensor,
                             target_len: int,
                             pad_mode: str = 'constant',
                             pad_value: typing.Any = 0,
                             take_first: bool = True) -> tf.Tensor:
        sequence_len = tf.shape(sequence)[0]
        diff_len = abs(target_len - sequence_len)
        if sequence_len < target_len:
            pad1 = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
            pad2 = diff_len - pad1
            sequence = tf.pad(sequence,
                              paddings=[[pad1, pad2]],
                              mode=pad_mode,
                              constant_values=pad_value)
        elif sequence_len > target_len:
            if take_first:
                sequence = sequence[:target_len]
            else:
                idx = tf.random.uniform([], maxval=diff_len, dtype=tf.int32)
                sequence = sequence[idx:(idx + target_len)]
        return tf.reshape(sequence, [target_len])

    def process_label(self, label: str, max_label_len: int = MAX_AUDIO_LEN) -> tf.Tensor:
        label = tf.strings.lower(label, 'utf-8')
        label = tf.strings.regex_replace(label, " ", self.space_token)
        label = tf.strings.unicode_split(label, input_encoding='UTF-8')
        label = tf.cond(
            tf.shape(label) < max_label_len,
            lambda: tf.pad(label, [[0, max_label_len - tf.shape(label)[0]]],
                           constant_values=self.pad_token),
            lambda: tf.slice(label, [0], [max_label_len]))
        label = tf.pad(label, [[1, 0]], constant_values=self.sos_token)
        label = tf.pad(label, [[0, 1]], constant_values=self.eos_token)
        label = self.char_to_num(label)
        return label

    def process_example_with_labels(self, audio_path: str, label: str) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        return self.process_example(audio_path), self.process_label(label)

    def process_example(self, audio_path: str) -> tf.Tensor:
        audio = self.decode_audio(audio_path)
        audio = self.crop_or_pad_sequence(audio, self.MAX_AUDIO_LEN,
                                          self.pad_mode, self.take_first)
        spectrogram = tf.signal.stft(audio,
                                     frame_length=self.win_length,
                                     frame_step=self.hop_length,
                                     fft_length=self.n_fft)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        return spectrogram

    def get_dataset(self) -> tf.data.Dataset:
        process_example_fn = self.process_example_with_labels if self.labels is not None else self.process_example
        if self.labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.audio_paths, self.labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.audio_paths)
        dataset = dataset.map(process_example_fn,
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset


if __name__ == '__main__':
    import pandas as pd

    p = '/app/datasets/audio_dataset/df.csv'
    df = pd.read_csv(p)
    df.path = df.path.apply(
        lambda x: os.path.join('/app/datasets/audio_dataset', x))
    assert Path(df.path[0]).is_file()
    dloader = LoadAudiosAndLabels(list(df.path), list(df.text))
    dloader.process_example(df.path[0])
    label = dloader.process_label('ewefwfw')
    print(type(label))
    # print(dloader.char_to_num.get_vocabulary())
    # print(dloader.char_to_num('*'))
    # print(dloader.num_to_char(0))
    print(list(df.text[:10]))
    for x in df.text[:10]:
        print(len(x))
    ds = dloader.get_dataset()
    for x in ds.take(10):
        print(dloader.num_to_char(x[1]))
