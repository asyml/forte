"""
File readers.
"""
import logging
from abc import abstractmethod, ABC
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


class MonoFileReader(PackReader, ABC):
    """Data reader that reads one data pack from each single text files.
    To be inherited by all mono file data readers.
    """

    # pylint: disable=no-self-use
    def _cache_key_function(self, file_directory: str):
        return file_directory.split('/')[-1]

    # pylint: disable=no-self-use
    def _collect(self, file_directory: str) -> Iterator[str]:  # type: ignore
        """
        :param file_directory: the path to a single directory containing the
        files.
        :return: Iterator[Any] collections to iterate over
        """
        return dataset_path_iterator(file_directory, "")

    @abstractmethod
    def parse_pack(self, files_path: str) -> DataPack:
        """
        Read a single DataPack from a file path.
        Args:
            files_path: The path to the file to read or information to parse.
        """
        raise NotImplementedError


class MonoFileMultiPackReader(MultiPackReader, ABC):
    """Data reader that reads one MultiPack from each single text files.
    """

    # pylint: disable=no-self-use
    def _cache_key_function(self, collection: str):
        return str(collection).split('/')[-1]

    # pylint: disable=no-self-use
    def _collect(self, data_path) -> Iterator[Any]:  # type: ignore
        """
        :param data_path: The datapath to the files.
        :return: Iterator[Any] collections to iterate over
        """
        return dataset_path_iterator(data_path, "")

    @abstractmethod
    def parse_pack(self, collection: str) -> MultiPack:
        """
        Read a MultiPack from a collection
        Args:
            collection: The path to the file to read or information to parse.
        """
        raise NotImplementedError
