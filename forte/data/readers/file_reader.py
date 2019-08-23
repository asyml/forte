"""
File readers.
"""
import logging
from typing import Iterator, Any

from forte.data.io_utils import dataset_path_iterator
from forte.data.multi_pack import MultiPack
from forte.data.readers.base_reader import PackReader, MultiPackReader
from forte.data.data_pack import DataPack

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

    def _collect(self, dir_path: str) -> Iterator[Any]:
        return dataset_path_iterator(dir_path, "")

    def parse_pack(self, collection: Any) -> DataPack:
        """
        Read a single datapack from a collection(file path in this case)
        Args:
            collection: The path to the file to read or information to parse.
        """
        pass


class MonoFileMultiPackReader(MultiPackReader):
    """Data reader that reads one MultiPack from each single text files.
    """

    def _collect(self, dir_path: str) -> Iterator[Any]:
        return dataset_path_iterator(dir_path, "")

    def parse_pack(self, collection: Any) -> MultiPack:
        """
        Read a single datapack from a collection(file path in this case)
        Args:
            collection: The path to the file to read or information to parse.
        """
        pass
