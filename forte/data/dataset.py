"""The Dataset class which deal with dataset level operation and store dataset
level meta.
"""
from typing import Dict, Union, Iterable, Optional, List, Type
from forte.data.data_pack import DataPack
from forte.data.ontology import Entry, Annotation

__all__ = [
    "Dataset",
]


class Dataset:
    """
    A wrapper of the dataset iterator.

    # currently only used to support texar batcher.
    """
    def __init__(self, dataset: Iterable[DataPack]):
        self.dataset: Iterable[DataPack] = dataset

    def get_data(
            self,
            context_type: Type[Annotation],
            requests: Optional[Dict[Type[Entry], Union[Dict, List]]] = None
    ):
        for pack in self.dataset:
            for data in pack.get_data(context_type, requests=requests):
                yield data
