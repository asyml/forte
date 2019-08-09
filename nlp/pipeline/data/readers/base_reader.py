"""
Base reader type to be inherited by all readers.
"""
from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Dict, Type, List, Union, Generic

import jsonpickle

from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.multi_pack import MultiPack
from nlp.pipeline.data.base_pack import PackType
from nlp.pipeline.data.ontology import Entry, base_ontology
from nlp.pipeline.utils import get_full_module_name

__all__ = [
    "BaseReader",
    "PackReader",
    'MultiPackReader'
]


class BaseReader(Generic[PackType]):
    """The basic data reader class.
    To be inherited by all data readers.
    """
    def __init__(self, lazy: bool = True) -> None:
        self.lazy = lazy
        self._cache_directory: Optional[Path] = None
        self._ontology = base_ontology
        self.output_info: Dict[Type[Entry], Union[List, Dict]] = {}
        self.component_name = get_full_module_name(self)

    def set_ontology(self, ontology):
        self._ontology = ontology
        self.define_output_info()

    @abstractmethod
    def define_output_info(self):
        pass

    @staticmethod
    def serialize_instance(instance: PackType) -> str:
        """
        Serializes a pack to a string.
        """
        return jsonpickle.encode(instance, unpicklable=True)

    @staticmethod
    def deserialize_instance(string: str) -> PackType:
        """
        Deserializes an pack from a string.
        """
        return jsonpickle.decode(string)

    @abstractmethod
    def iter(self, dataset) -> Union[List[PackType], Iterator[PackType]]:
        """
        An iterator over the entire dataset, yielding all documents processed.
        Should call :meth:`read` to read each document.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, data):
        """
        Read and return one :class:`BasePack` object. Should update
        config.working_component at the begining and the end of this method.
        """
        raise NotImplementedError

    @abstractmethod
    def _instances_from_cache_file(self,
                                   cache_filename: Path):
        raise NotImplementedError

    def cache_data(self, cache_directory: str) -> None:
        """Specify the path to the cache directory.

        After you call this method, the dataset reader will use this
        :attr:`cache_directory` to store a cache of :class:`BasePack` read
        from every document passed to :func:`read`, serialized as one
        string-formatted :class:`BasePack`. If the cache file for a given
        ``file_path`` exists, we read the :class:`BasePack` from the cache
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

    def _record_fields(self, pack: PackType):
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
            pack.record_fields(fields, entry_type, component)


class PackReader(BaseReader[DataPack]):
    """The basic data reader class.
    To be inherited by all data readers.
    """

    def _instances_from_cache_file(self,
                                   cache_filename: Path) -> Iterator[DataPack]:
        with cache_filename.open("r") as cache_file:
            for line in cache_file:
                pack = self.deserialize_instance(line.strip())
                if not isinstance(pack, DataPack):
                    raise TypeError(f"Pack deserialized from {cache_filename} "
                                    f"is {type(pack)}, but expect {DataPack}")
                yield pack


class MultiPackReader(BaseReader[MultiPack]):
    """The basic MultiPack data reader class.
    To be inherited by all data readers which return MultiPack.
    """

    def _instances_from_cache_file(self,
                                   cache_filename: Path) -> Iterator[MultiPack]:
        with cache_filename.open("r") as cache_file:
            for line in cache_file:
                pack = self.deserialize_instance(line.strip())
                if not isinstance(pack, MultiPack):
                    raise TypeError(f"Pack deserialized from {cache_filename} "
                                    f"is {type(pack)}, but expect {MultiPack}")
                yield pack
