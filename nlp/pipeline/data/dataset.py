"""The Dataset class which deal with dataset level operation and store dataset
level meta.
"""
from typing import Dict, Union, Iterable, Optional, List
from nlp.pipeline.data.data_pack import DataPack

__all__ = [
    "Dataset",
]


class Dataset:
    def __init__(self, dataset: Iterable[DataPack]):
        self.dataset: Iterable[DataPack] = dataset

    def get_data(
            self,
            context_type: str,
            annotation_types: Optional[Dict[str, Union[Dict, List]]] = None,
            link_types: Optional[Dict[str, Union[Dict, List]]] = None,
            group_types: Optional[Dict[str, Union[Dict, List]]] = None):
        for pack in self.dataset:
            for data in pack.get_data(context_type, annotation_types,
                                      link_types, group_types):
                yield data
