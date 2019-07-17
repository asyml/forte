"""The Dataset class which deal with dataset level operation and store dataset
level meta.
"""
from typing import Iterator, Dict, Union, Iterable
from nlp.pipeline.data.data_pack import DataPack

__all__ = [
    "Dataset",
]


class Dataset:
    def __init__(self, dataset: Iterable[DataPack]):
        self.dataset: Iterable[DataPack] = dataset

    def get_data(self,
                 context_type: str,
                 annotation_types: Dict[str, Union[Dict, Iterable]] = None,
                 link_types: Dict[str, Union[Dict, Iterable]] = None,
                 group_types: Dict[str, Union[Dict, Iterable]] = None):
        for pack in self.dataset:
            for data in pack.get_data(context_type, annotation_types,
                                      link_types, group_types):
                yield data


