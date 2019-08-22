"""
File readers.
"""
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Union, Any

from forte import config
from forte.data.io_utils import dataset_path_iterator
from forte.data.multi_pack import MultiPack
from forte.data.readers.base_reader import PackReader, MultiPackReader
from forte.data.data_pack import DataPack, ReplaceOperationsType

logger = logging.getLogger(__name__)

__all__ = [
    "MonoFileReader",
    "MonoFileMultiPackReader",
    "PackReader"
]


class MonoFileReader(PackReader):
    """Data reader that reads one data pack from each single text files.
    To be inherited by all mono file data readers.
    """

    @staticmethod
    def collect(dir_path: str) -> Iterator[Any]:  # type: ignore
        return dataset_path_iterator(dir_path, "")

    def parse_pack(self, collection: Any,
                   replace_operations: Optional[ReplaceOperationsType]
                   ) -> DataPack:
        """
        Read a single datapack from a collection(file path in this case)
        Args:
            collection: The path to the file to read or information to parse.
            replace_operations (ReplaceOperationsType, optional): A list of
                operations, where each operation is in the form of a tuple with
                the values - (1) span or a regex to be replaced (2) the
                corresponding replacement string.
        """
        pass


class MonoFileMultiPackReader(MultiPackReader):
    """Data reader that reads one MultiPack from each single text files.
    """

    @staticmethod
    def collect(dir_path: str) -> Iterator[Any]:  # type: ignore
        return dataset_path_iterator(dir_path, "")

    def parse_pack(self, collection: Any,
                   replace_operations: Optional[ReplaceOperationsType]
                   ) -> MultiPack:
        """
        Read a single datapack from a collection(file path in this case)
        Args:
            collection: The path to the file to read or information to parse.
            replace_operations (ReplaceOperationsType, optional): A list of
                operations, where each operation is in the form of a tuple with
                the values - (1) span or a regex to be replaced (2) the
                corresponding replacement string.
        """
        pass

    @abstractmethod
    def _read_document(self, file_path: str):
        """
        Process the original document. Should be Implemented according to the
        document formant.
        """
        raise NotImplementedError
