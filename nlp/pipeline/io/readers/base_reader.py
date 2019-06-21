"""
Base reader type to be inherited by all readers.
"""
import os
import pathlib
from typing import Iterator
import jsonpickle
from nlp.pipeline.io.data_pack import DataPack


class BaseReader:
    """The basic data reader class.
    To be inherited by all data readers.
    """
    def __init__(self, lazy: bool = True) -> None:
        self.lazy = lazy
        self._cache_directory = None
        self.component_name = f"{__name__}.{self.__class__.__name__}"
        self.current_datapack = None

    def cache_data(self, cache_directory: str) -> None:
        """Specify the path to the cache directory.

        After you call this method, the dataset reader will use this
        :attr:`cache_directory` to store a cache of already-processed
        :class:`DataPack` in every document passed to :func:`read`,
        serialized as one string-formatted :class:`DataPack`. If the cache
        file for a given ``file_path`` exists, we read the :class:`DataPack`
        from the cache instead of re-processing the data (using
        :func:`deserialize_instance`).  If the cache file does not
        exist, we will `create` it on our first pass through the data
        (using :func:`serialize_instance`).

        """
        self._cache_directory = pathlib.Path(cache_directory)
        os.makedirs(self._cache_directory, exist_ok=True)

    def _get_cache_location_for_file_path(self, file_path: str) -> str:
        return f"{self._cache_directory / file_path.split('/')[-1]}.cache"

    def _instances_from_cache_file(self, cache_filename: str) \
            -> Iterator[DataPack]:
        with open(cache_filename, "r") as cache_file:
            for line in cache_file:
                yield self.deserialize_instance(line.strip())

    @staticmethod
    def serialize_instance(instance: DataPack) -> str:
        """
        Serializes an ``DataPack`` to a string.
        """
        return jsonpickle.encode(instance)

    @staticmethod
    def deserialize_instance(string: str) -> DataPack:
        """
        Deserializes an ``DataPack`` from a string.
        """
        return jsonpickle.decode(string)

    def dataset_iterator(self, *args):
        """
        An iterator over the entire dataset, yielding all documents processed.
        Should call :meth:`read` to read each document.
        """
        raise NotImplementedError

    def read(self, *args) -> DataPack:
        """
        Read a **single** document from the dataset.
        """
        raise NotImplementedError
