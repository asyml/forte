"""
File readers.
"""
import logging
from abc import abstractmethod
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

    # pylint: disable=no-self-use
    def _cache_key_function(self, collection):
        return str(collection).split('/')[-1]

    # pylint: disable=no-self-use
    def _collect(self, **kwargs) -> Iterator[Any]:
        """
        May or maynot be reimplemented by a child reader class
        :param kwargs: Expecting the data_source keyword as default
        :return: Iterator[Any] collections to iterate over
        """
        return dataset_path_iterator(kwargs['data_source'], "")

    @abstractmethod
    def parse_pack(self, collection: Any) -> DataPack:
        """
        Read a single DataPack from a collection
        Args:
            collection: The path to the file to read or information to parse.
        """
        raise NotImplementedError


class MonoFileMultiPackReader(MultiPackReader):
    """Data reader that reads one MultiPack from each single text files.
    """

    # pylint: disable=no-self-use
    def _cache_key_function(self, collection):
        return str(collection).split('/')[-1]

    # pylint: disable=no-self-use
    def _collect(self, **kwargs) -> Iterator[Any]:
        """
        May or maynot be reimplemented by a child reader class
        :param kwargs: Expecting the data_source keyword as default
        :return: Iterator[Any] collections to iterate over
        """
        return dataset_path_iterator(kwargs['data_source'], "")

    @abstractmethod
    def parse_pack(self, collection: Any) -> MultiPack:
        """
        Read a MultiPack from a collection
        Args:
            collection: The path to the file to read or information to parse.
        """
        raise NotImplementedError
