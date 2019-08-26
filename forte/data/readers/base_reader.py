"""
Base reader type to be inherited by all readers.
"""
from abc import abstractmethod
from pathlib import Path
from typing import (Iterator, Optional, Dict, Type, List, Union, Generic,
                    Any)
import os
import logging
import jsonpickle

from forte import config
from forte.data.data_pack import DataPack, ReplaceOperationsType
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
                 append_to_cache: bool = False):
        """
        Args:
        lazy (bool): If lazy is true, will use a Iterator to iterate
            collections. If False, the reader will use a list of collections
        from_cache (bool, optional): Decide whether to read from cache
            if cache file exists. By default (``True``), the reader will
            try to read an datapack from the first line of the caching file.
            If ``False``, the reader will only read from the original file
            and use the cache file path only for output.
        cache_file (str, optional): The path of the caching file. If
            :attr:`cache_file_path` is ``None`` and
            :attr:`self._cache_directory` is not ``None``, use the result
            of :meth:`_get_cache_location_for_file_path`. If both
            :attr:`cache_file_path` and :attr:`self._cache_directory`
            are ``None``, will not read from or write to a caching file.
        append_to_cache (bool, optional): Decide whether to append write
            if cache file already exists.  By default (``False``), we
            will overwrite the existing caching file. If ``True``, we will
            cache the datapack append to end of the caching file.
    """

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

    @abstractmethod
    def _collect(self, **kwargs) -> Iterator[Any]:
        """
        Gives an iterator of data objects
        each individual object should contain sufficient information
        needed to construct or locate a data pack in cache.
        For example: `data_source` can be a kwarg which is the path to a file
                     that a reader can take to read and parse a file.
        :param: One of more data sources
        :return Iterator[Any]: Implementation should yield a collection
        """
        raise NotImplementedError

    @abstractmethod
    def parse_pack(self, collection: Any) -> PackType:
        """
        Gives an iterator of Packs parsed from a collection
        :param collection: Object that can be parsed into a Pack
        :return Iterator[PackType]: Iterator of Packs
        """
        raise NotImplementedError

    def text_replace_operation(self, text: str) -> ReplaceOperationsType:
        """
        Given the possibly noisy text, compute and return the
        replacement operations in the form of a list of (span, str)
        pairs, where the content in the span will be replaced by the
        corresponding str.
        :param text: The original data text to be cleaned.
        :return List[Tuple[Tuple[int, int], str]]: the replacement operations
        """
        return []

    @abstractmethod
    def _cache_key_function(self, collection: Any) -> str:
        """
        Computes the cache key based on the type of data
        :param collection:  Any object that provides information
         to identify the name and location of the cache file
        :return: str that specifies the path to cache file
        """
        raise NotImplementedError

    def _get_cache_location(self, collection: Any) -> Path:
        """
        Gets the path to the cache file for a collection
        :param collection: information to compute cache key
        :return Path: file path to the cache file for a Pack
        """
        file_path = self._cache_key_function(collection)
        return Path(os.path.join(str(self._cache_directory), file_path))

    def read_from_cache(self, cache_location) -> PackType:
        """
        Reads one or more Packs from a cache_location
        :param cache_location: Path to the cache file
        :return: Pack
        """
        logger.info("reading from cache file %s", cache_location)
        return next(self._instances_from_cache_file(cache_location))

    def iter(self, **kwargs) -> Union[Iterator[PackType], List[PackType]]:
        """
        An iterator over the entire dataset, giving all Packs processed
         as list or Iterator depending on `lazy`, giving all the Packs read
         from the data source(s). If not reading from cache,
         should call collect()
        :param kwargs: One or more input data sources
        for example, most DataPack readers
        accept `data_source` as file/folder path
        :return: Either Iterator or List depending on setting of `lazy`
        """
        if self.lazy:
            return self._lazy_iter(**kwargs)

        else:
            datapacks: List[PackType] = []

            for collection in self._collect(**kwargs):
                if self.from_cache:
                    pack = self.read_from_cache(
                        self._get_cache_location(collection))
                else:

                    pack = self.parse_pack(collection)

                    # write to the cache if we need to.
                    if self.cache_file:
                        self.cache_data(self.cache_file, pack)

                    self._record_fields(pack)
                    if not isinstance(pack, DataPack):
                        raise ValueError(
                            f"No Pack object read from the given "
                            f"collection {collection}, returned {type(pack)}."
                        )
                config.working_component = None
                datapacks.append(pack)
            return datapacks

    def _lazy_iter(self, **kwargs):
        for collection in self._collect(**kwargs):
            if self.from_cache:
                pack = self.read_from_cache(
                    self._get_cache_location(collection))
            else:
                pack = self.parse_pack(collection)

                # write to the cache if we need to.
                if self.cache_file:
                    self.cache_data(self.cache_file, pack)

                self._record_fields(pack)
                if not isinstance(pack, DataPack):
                    raise ValueError(
                        f"No Pack object read from the given "
                        f"collection {collection}, returned {type(pack)}."
                    )
            yield pack

    def cache_data(self, cache_directory: Path,
                   pack: PackType) -> None:
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
        self._cache_directory = cache_directory
        Path.mkdir(self._cache_directory, exist_ok=True)
        logger.info("Caching pack to %s", self.cache_file)
        if self.append_to_cache:
            with self.cache_file.open('a') as cache:  # type: ignore
                cache.write(self.serialize_instance(pack) + "\n")
        else:
            with self.cache_file.open('w') as cache:  # type: ignore
                cache.write(self.serialize_instance(pack) + "\n")

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
                                    f"is {type(pack)},"
                                    f"but expect {self.pack_type}")
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
