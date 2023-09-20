import jiwer
import tensorflow as tf

from asr.models.ctc_decoder import CTCDecoder


class WordErrorRate(tf.keras.metrics.Metric):

    def __init__(self, decoder: CTCDecoder, name='word_error_rate', **kwargs):
        super(WordErrorRate, self).__init__(name=name, **kwargs)
        self.decoder = decoder
        self.num_to_char = decoder.num_to_char
        self.reset_state()

    def update_state(self, y_true, y_pred, sample_weight=None):
        predictions = self.decoder.decode_predictions(y_pred)
        labels = tf.vectorized_map(
            lambda x: tf.strings.reduce_join(self.num_to_char(x)), y_true)
        self.value.assign_add(
            tf.py_function(self._wer_numpy, [labels, predictions], tf.float32))
        self.n_batches.assign_add(1)

    def _wer_numpy(self, labels, predictions):
        labels = [x.decode('utf-8') for x in labels.numpy()]
        predictions = [x.decode('utf-8') for x in predictions.numpy()]
        return jiwer.wer(labels, predictions)

    def result(self):
        return tf.divide(self.value, self.n_batches)

    def reset_state(self):
        self.value = tf.Variable(0, dtype=tf.float32)
        self.n_batches = tf.Variable(0, dtype=tf.float32)
