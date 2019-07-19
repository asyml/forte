"""
Base reader type to be inherited by all readers.
"""
from pathlib import Path
from typing import Iterator, Optional

import jsonpickle

from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.utils import get_full_component_name

__all__ = [
    "BaseReader",
]


class BaseReader:
    """The basic data reader class.
    To be inherited by all data readers.
    """
    def __init__(self, lazy: bool = True) -> None:
        self.lazy = lazy
        self._cache_directory: Optional[Path] = None
        self.component_name = get_full_component_name(self)

    def cache_data(self, cache_directory: str) -> None:
        """Specify the path to the cache directory.

        After you call this method, the dataset reader will use this
        :attr:`cache_directory` to store a cache of :class:`DataPack` read
        from every document passed to :func:`read`, serialized as one
        string-formatted :class:`DataPack`. If the cache file for a given
        ``file_path`` exists, we read the :class:`DataPack` from the cache
        (using :func:`deserialize_instance`).  If the cache file does not
        exist, we will `create` it on our first pass through the data (using
        :func:`serialize_instance`).
        """
        self._cache_directory = Path(cache_directory)
        Path.mkdir(self._cache_directory, exist_ok=True)

    def _get_cache_location_for_file_path(self,
                                          file_path: str) -> Optional[Path]:
        if self._cache_directory is None:
            return None
        while file_path[-1] == '/':
            file_path = file_path[:-1]
        return Path(f"{self._cache_directory / file_path.split('/')[-1]}.cache")

    def _instances_from_cache_file(self,
                                   cache_filename: Path) -> Iterator[DataPack]:
        with cache_filename.open("r") as cache_file:
            for line in cache_file:
                yield self.deserialize_instance(line.strip())

    @staticmethod
    def serialize_instance(instance: DataPack, unpicklable: bool = True) -> str:
        """
        Serializes an ``DataPack`` to a string.
        """
        return jsonpickle.encode(instance, unpicklable=unpicklable)

    @staticmethod
    def deserialize_instance(string: str) -> DataPack:
        """
        Deserializes an ``DataPack`` from a string.
        """
        return jsonpickle.decode(string)

    def dataset_iterator(self, *args, **kwargs):
        """
        An iterator over the entire dataset, yielding all documents processed.
        Should call :meth:`read` to read each document.
        """
        raise NotImplementedError

    def read(self, *args, **kwargs) -> DataPack:
        """
        Read a **single** document from the dataset.
        """
        raise NotImplementedError
