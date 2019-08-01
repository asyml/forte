"""
Base reader type to be inherited by all readers.
"""
from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Dict, Type, List, Union

import jsonpickle

from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.ontology import Entry
from nlp.pipeline.utils import get_full_module_name

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
        self._ontology = None
        self.output_info: Dict[Type[Entry], Union[List, Dict]] = {}
        self.component_name = get_full_module_name(self)
        self.current_datapack: DataPack = DataPack()

    def set_ontology(self, ontology):
        self._ontology = ontology
        self.define_output_info()

    @abstractmethod
    def define_output_info(self):
        pass

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
    def serialize_instance(instance: DataPack) -> str:
        """
        Serializes an ``DataPack`` to a string.
        """
        return jsonpickle.encode(instance, unpicklable=True)

    @staticmethod
    def deserialize_instance(string: str) -> DataPack:
        """
        Deserializes an ``DataPack`` from a string.
        """
        return jsonpickle.decode(string)

    def dataset_iterator(self, dataset):
        """
        An iterator over the entire dataset, yielding all documents processed.
        Should call :meth:`read` to read each document.
        """
        raise NotImplementedError

    def read(self, data) -> DataPack:
        """
        Read and return one Datapack. Should update config.working_component
        at the begining and the end of this method.
        """
        raise NotImplementedError

    def _record_fields(self):
        """
        Record the fields and entries that this processor add to packs.
        """
        for entry_type, info in self.output_info.items():
            component = self.component_name
            fields: List[str] = []
            if isinstance(info, list):
                fields = info
            elif isinstance(info, dict):
                fields = info["fields"]
                if "component" in info.keys():
                    component = info["component"]
            self.current_datapack.record_fields(fields, entry_type, component)
