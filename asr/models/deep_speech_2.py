import tensorflow as tf

from asr import utils


class DeepSpeech2(tf.keras.Model):

    def __init__(self,
                 input_dim,
                 output_dim,
                 rnn_layers=5,
                 rnn_units=128,
                 *args,
                 **kwargs):
        super(DeepSpeech2, self).__init__(*args, **kwargs)
        self._model = self._build_model(input_dim, output_dim, rnn_layers,
                                        rnn_units)

    def load_checkpoint(self,
                        checkpoint_path,
                        skip_mismatch=False,
                        by_name=False,
                        options=None,
                        logger=None):
        return utils.load_checkpoint(self._model,
                                     checkpoint_path,
                                     skip_mismatch=skip_mismatch,
                                     by_name=by_name,
                                     options=options,
                                     logger=logger)

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs, training=training, mask=mask)

    def _build_model(self, input_dim, output_dim, rnn_layers, rnn_units):
        input_spectrogram = tf.keras.layers.Input((None, input_dim),
                                                  name='input')
        # Expand the dimension to use 2D CNN.
        x = tf.keras.layers.Reshape((-1, input_dim, 1),
                                    name='expand_dim')(input_spectrogram)
        # Convolution layer 1
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[11, 41],
            strides=[2, 2],
            padding='same',
            use_bias=False,
            name='conv_1',
        )(x)
        x = tf.keras.layers.BatchNormalization(name='conv_1_bn')(x)
        x = tf.keras.layers.ReLU(name='conv_1_relu')(x)
        # Convolution layer 2
        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[11, 21],
            strides=[1, 2],
            padding='same',
            use_bias=False,
            name='conv_2',
        )(x)
        x = tf.keras.layers.BatchNormalization(name='conv_2_bn')(x)
        x = tf.keras.layers.ReLU(name='conv_2_relu')(x)
        # Reshape the resulted volume to feed the RNNs layers
        x = tf.keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        # RNN layers
        for i in range(1, rnn_layers + 1):
            recurrent = tf.keras.layers.GRU(
                units=rnn_units,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f'gru_{i}',
            )
            x = tf.keras.layers.Bidirectional(recurrent,
                                              name=f'bidirectional_{i}',
                                              merge_mode='concat')(x)
            if i < rnn_layers:
                x = tf.keras.layers.Dropout(rate=0.5)(x)
        # Dense layer
        x = tf.keras.layers.Dense(units=rnn_units * 2, name='dense_1')(x)
        x = tf.keras.layers.ReLU(name='dense_1_relu')(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        # Classification layer
        output = tf.keras.layers.Dense(units=output_dim + 1,
                                       activation='softmax')(x)
        # Model
        model = tf.keras.Model(input_spectrogram, output, name='DeepSpeech_2')
        return model
