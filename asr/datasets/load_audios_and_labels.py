import typing
from typing import Optional, Sequence, Tuple

import tensorflow as tf
import tensorflow_io as tfio

from asr.datasets import base_dataloaders, common


class AudiosAndLabelsLoader(base_dataloaders.TextDataLoader):
    MAX_AUDIO_LEN: int = 352960
    MAX_LABEL_LEN: int = 180

    SOS_TOKEN: str = '<sos>'
    EOS_TOKEN: str = '<eos>'
    PAD_TOKEN: str = '<pad>'
    SPACE_TOKEN: str = '|'
    OOV_TOKEN: str = '*'

    def __init__(self,
                 audio_paths: Sequence[str],
                 labels: Optional[Sequence[str]] = None,
                 sample_rate: int = 16000,
                 pad_mode: str = 'constant',
                 take_first: bool = True,
                 spec_shape: Tuple[int, int] = (384, 512),
                 win_length: Optional[int] = None,
                 n_fft: int = 2048,
                 f_min: int = 0,
                 f_max: int = 8000,
                 language: str = 'russian',
                 max_label_len: int = MAX_LABEL_LEN,
                 sos_token: str = SOS_TOKEN,
                 eos_token: str = EOS_TOKEN,
                 pad_token: str = PAD_TOKEN,
                 space_token: str = SPACE_TOKEN,
                 oov_token: str = OOV_TOKEN,
                 batch_size: int = 8,
                 drop_remainder: bool = False,
                 cache: bool = False,
                 shuffle: Optional[int] = 64,
                 seed: Optional[int] = 43):
        super(AudiosAndLabelsLoader, self).__init__()
        self.audio_paths = audio_paths
        if labels is not None:
            language_enum = common.STR_LANGUAGE_TO_ENUM_MAP.get(language, None)
            if language_enum is None:
                supported_languages = list(
                    common.STR_LANGUAGE_TO_ENUM_MAP.keys())
                raise ValueError(
                    f'Language "{language}" is not supported. Supported languages: {supported_languages}'
                )
            vocabulary = common.LANGUAGE_TO_VOCABULARY_MAP[language_enum]
            vocabulary += (sos_token, eos_token, pad_token, space_token)
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=vocabulary, oov_token=oov_token)
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(),
                oov_token=oov_token,
                invert=True)
        self.labels = labels
        self.sample_rate = sample_rate
        self.pad_mode = pad_mode
        self.take_first = take_first
        self.spec_shape = spec_shape
        self.win_length = win_length or n_fft
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.max_label_len = max_label_len
        self.eos_token = eos_token
        self.sos_token = sos_token
        self.pad_token = pad_token
        self.space_token = space_token
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        self.cache = cache
        self.shuffle = shuffle
        self.seed = seed

    def get_dataset(self) -> tf.data.Dataset:
        process_example_fn = self._process_example_with_labels if self.labels is not None else self._process_example
        if self.labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.audio_paths, self.labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.audio_paths)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.with_options(self._dataset_options)
        if self.shuffle is not None:
            dataset = dataset.shuffle(self.shuffle, seed=self.seed)
        dataset = dataset.map(process_example_fn,
                              num_parallel_calls=tf.data.AUTOTUNE)
        if self.cache:
            dataset = dataset.cache()
        dataset = dataset.prefetch(self.batch_size)
        dataset = dataset.batch(self.batch_size,
                                drop_remainder=self.drop_remainder)
        return dataset

    def _process_example_with_labels(
            self, audio_path: str,
            label: str) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        return self._process_example(audio_path), self._process_label(label)

    def _process_example(self,
                         audio_path: str,
                         max_audio_len: int = MAX_AUDIO_LEN,
                         top_db: int = 80) -> tf.Tensor:
        audio = self._decode_audio(audio_path, self.sample_rate)
        audio = self._crop_or_pad_sequence(audio, max_audio_len, self.pad_mode,
                                           self.take_first)
        audio_len = tf.shape(audio)[0]
        hop_length = tf.cast((audio_len // (self.spec_shape[1] - 1)), tf.int32)
        spec = tfio.audio.spectrogram(audio,
                                      nfft=self.n_fft,
                                      window=self.win_length,
                                      stride=hop_length)
        mel_spec = tfio.audio.melscale(spec,
                                       rate=self.sample_rate,
                                       mels=self.spec_shape[0],
                                       fmin=self.f_min,
                                       fmax=self.f_max)
        db_mel_spec = tfio.audio.dbscale(mel_spec, top_db=top_db)
        db_mel_spec = tf.transpose(db_mel_spec, perm=[1, 0])
        means = tf.math.reduce_mean(db_mel_spec, 1, keepdims=True)
        stddevs = tf.math.reduce_std(db_mel_spec, 1, keepdims=True)
        norm_db_mel_spec = tf.math.divide_no_nan(db_mel_spec - means,
                                                 stddevs + 1e-10)
        return norm_db_mel_spec

    def _process_label(self,
                       label: str,
                       max_label_len: int = MAX_LABEL_LEN) -> tf.Tensor:
        label = tf.strings.lower(label, 'utf-8')
        label = tf.strings.regex_replace(label, " ", self.space_token)
        label = tf.strings.unicode_split(label, input_encoding='UTF-8')
        label = tf.cond(
            tf.shape(label)[0] < max_label_len,
            lambda: tf.pad(label, [[0, max_label_len - tf.shape(label)[0]]],
                           constant_values=self.pad_token),
            lambda: tf.slice(label, [0], [max_label_len]))
        label = tf.pad(label, [[1, 0]], constant_values=self.sos_token)
        label = tf.pad(label, [[0, 1]], constant_values=self.eos_token)
        label = self.char_to_num(label)
        return label

    def _decode_audio(self, audio_path: str,
                      target_sample_rate: int) -> tf.Tensor:
        audio_bytes = tf.io.read_file(audio_path)
        audio, sample_rate = tf.audio.decode_wav(audio_bytes)
        audio = tfio.audio.resample(audio, tf.cast(sample_rate, tf.int64),
                                    target_sample_rate)
        # stereo to mono
        if tf.shape(audio)[1] > 1:
            audio = audio[:, 0]
        audio = tf.squeeze(audio, -1)
        return audio

    def _crop_or_pad_sequence(self,
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

    @property
    def _dataset_options(self) -> tf.data.Options:
        options = tf.data.Options()
        options.deterministic = False
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
        return options
