import traceback
from typing import Dict

from asr.datasets.audio_dataset_dataloader import AudioDatasetLoader
from asr.datasets.base_dataloaders import BaseDataLoader
from asr.datasets.load_audios_and_labels import AudiosAndLabelsLoader


def get_dataloader(config: Dict) -> BaseDataLoader:
    dataloader_name = config['type']
    dataloader_params = config.get('params', {})
    try:
        dataloader_type = eval(dataloader_name)
    except NameError as err:
        traceback.print_tb(err.__traceback__)
        raise ValueError(f'Undefined dataloader type {dataloader_name}.')
    dataloader = dataloader_type(**dataloader_params)
    return dataloader
