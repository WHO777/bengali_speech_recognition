from pathlib import Path

import pandas as pd
import tensorflow as tf

from asr.datasets import load_audios_and_labels


class AudioDatasetLoader(load_audios_and_labels.LoadAudiosAndLabels):

    def __init__(self, root_dir: str, *args, **kwargs):
        df_path = Path(root_dir) / 'df.csv'
        if not df_path.is_file():
            raise RuntimeError(f'There is not file "df.csv" in {root_dir}')
        df = pd.read_csv(df_path)

        def abs_path_fn(path):
            if not Path(path).is_absolute():
                path = str(Path(root_dir) / path)
            return path

        audio_paths = list(df.path.apply(abs_path_fn))
        labels = list(df.text)

        super(AudioDatasetLoader, self).__init__(audio_paths=audio_paths,
                                                 labels=labels,
                                                 *args,
                                                 **kwargs)

    def get_dataset(self) -> tf.data.Dataset:
        return super().get_dataset()
