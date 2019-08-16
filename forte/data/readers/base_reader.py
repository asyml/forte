"""
Base reader type to be inherited by all readers.
"""
from abc import abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Dict, Type, List, Union, Generic

import jsonpickle
import logging

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
    def __init__(self, lazy: bool = True) -> None:
        self.lazy = lazy
        self._cache_directory: Optional[Path] = None
        self._ontology = base_ontology
        self.output_info: Dict[Type[Entry], Union[List, Dict]] = {}
        self.component_name = get_full_module_name(self)

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

    @abstractmethod
    def iter(self, **kwargs) -> Union[List[PackType], Iterator[PackType]]:
        """
        An iterator over the entire dataset, yielding all Packs processed.
        If not reading from cache, should call collect()
        """
        cache_file = None
        if self._cache_directory is not None:
            cache_file = self._get_cache_location_for_file_path(dir_path)

        has_cache = cache_file is not None and cache_file.exists()

        if has_cache:
            logger.info("reading from cache file %s", cache_file)
            return List(self._instances_from_cache_file(cache_file))

        logger.info("reading from original files in %s", dir_path)\


        if self.lazy:
            for collection in self.collect(**kwargs):
                yield self.parse_packs(collection)
        else:

            datapacks: List[DataPack] = []

            for collection in self.collect(**kwargs):
                datapacks.append(self.parse_packs(collection))
            return datapacks


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


    @abstractmethod
    def _cache_key_function(collection):
        # Computes the cache key based on the type of data
        # To be implemented by the reader
        raise NotImplementedError

    # TODO:
    def _get_cache_location(self, collection):
        """

        :param collect: information to compute cache key
        :return:
        """
        raise NotImplementedError

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