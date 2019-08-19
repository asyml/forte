"""
Base reader type to be inherited by all readers.
"""
from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Dict, Type, List, Union, Generic, Any

import jsonpickle
import logging
import os

from forte import config
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.base_pack import PackType
from forte.data.ontology import Entry, base_ontology
from forte.utils import get_full_module_name

__all__ = [
    "BaseReader",
    "PackReader",
    'MultiPackReader'
]

logger = logging.getLogger(__name__)

class BaseReader(Generic[PackType]):
    """The basic data reader class.
    To be inherited by all data readers.
    """
    def __init__(self,
                 lazy: bool = True,
                 from_cache: bool = False,
                 cache_file: Optional[Path] = None,
                 append_to_cache: bool = False)-> None:
        self.lazy = lazy
        self.from_cache = from_cache
        self._cache_directory: Optional[Path] = None
        self._ontology = base_ontology
        self.output_info: Dict[Type[Entry], Union[List, Dict]] = {}
        self.component_name = get_full_module_name(self)
        self.cache_file = cache_file
        self.append_to_cache = append_to_cache


    @property
    def pack_type(self):
        raise NotImplementedError

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

    def collect(self, **kwargs) -> Iterator[Any]:
        """
        Gives an iterator of some object that can be converted to a DataPack
        or indicate the cache key to read
        TO BE IMPLEMENTED BY THE READER
        :param: One of more data sources
        :return Iterator[Any]: Implementation should yield a collection
        """
        raise NotImplementedError

    def parse_pack(self, collection: Any) -> Iterator[PackType]:
        """
        Gives an iterator of Packs parsed from a collection
        TO BE IMPLEMENTED BY THE READER
        :param collection: Object that can be parsed into a Pack
        :return Iterator[PackType]: Iterator of Packs
        """
        raise NotImplementedError

    @abstractmethod
    def _cache_key_function(collection):
        # Computes the cache key based on the type of data
        # To be implemented by the reader
        raise NotImplementedError

    # TODO:
    def _get_cache_location(self, collection: Any) -> Path:
        """
        Gets the path to the cache file for a collection
        :param collection: information to compute cache key
        :return Path: file path to the cache file for a Pack
        """
        file_path = self._cache_key_function(collection)
        return os.path.join(self._cache_directory, file_path)

    def read_from_cache(self, cache_location) -> PackType:
        """

        :param cache_location: Path to the cache file
        :return: Pack
        """
        logger.info("reading from cache file %s", cache_location)
        return self._instances_from_cache_file(cache_location)


    def iter(self, **kwargs):# -> Union[Iterator[PackType], List[PackType]]:
        """
        An iterator over the entire dataset, yielding all Packs processed.
        If not reading from cache, should call collect()
        """
        if self.lazy:
            return self._lazy_iter(**kwargs)

        else:
            datapacks: List[DataPack] = []

            for collection in self.collect(**kwargs):
                if self.from_cache:
                    pack = self.read_from_cache(self._get_cache_location(collection))
                else:
                    pack = self.parse_pack(collection)

                    # write to the cache if we need to.
                    if self.cache_file:
                        logger.info("Caching datapack to %s", self.cache_file)
                        if self.append_to_cache:
                            with self.cache_file.open('a') as cache:
                                cache.write(self.serialize_instance(pack) + "\n")
                        else:
                            with self.cache_file.open('w') as cache:
                                cache.write(self.serialize_instance(pack) + "\n")

                    config.working_component = None
                    datapacks.append(pack)
            return datapacks

    def _lazy_iter(self, **kwargs):
        for collection in self.collect(**kwargs):
            if self.from_cache:
                pack = self.read_from_cache(self._get_cache_location(collection))
            else:
                pack = self.parse_pack(collection)
            yield pack

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

    def _instances_from_cache_file(self,
                                   cache_filename: Path) -> Iterator[PackType]:
        with cache_filename.open("r") as cache_file:
            for line in cache_file:
                pack = self.deserialize_instance(line.strip())
                if not isinstance(pack, self.pack_type):
                    raise TypeError(f"Pack deserialized from {cache_filename} "
                                    f"is {type(pack)}, but expect {self.pack_type}")
                yield pack


class PackReader(BaseReader[DataPack]):
    """The basic data reader class.
    To be inherited by all data readers.
    """

    @property
    def pack_type(self):
        return DataPack


class MultiPackReader(BaseReader[MultiPack]):
    """The basic MultiPack data reader class.
    To be inherited by all data readers which return MultiPack.
    """

    @property
    def pack_type(self):
        return MultiPack