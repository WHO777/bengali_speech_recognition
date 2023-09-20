from typing import List

import tensorflow as tf


class CTCDecoder:

    def __init__(self, num_to_char_lookup: tf.keras.layers.StringLookup):
        self.num_to_char = num_to_char_lookup

    def decode_predictions(self, pred: tf.Tensor) -> List[str]:
        input_len = tf.ones(tf.shape(pred)[0]) * tf.cast(
            tf.shape(pred)[1], tf.float32)
        # Use greedy search. For complex tasks, you can use beam search
        results = tf.keras.backend.ctc_decode(pred,
                                              input_length=input_len,
                                              greedy=True)[0][0]
        # Iterate over the results and get back the text
        output_text = tf.vectorized_map(
            lambda x: tf.strings.reduce_join(self.num_to_char(x)), results)
        return output_text
