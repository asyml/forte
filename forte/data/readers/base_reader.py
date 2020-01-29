# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base reader type to be inherited by all readers.
"""
import logging
import os
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Iterator, Optional, Any

import jsonpickle

from forte.common.resources import Resources
from forte.common.types import ReplaceOperationsType
from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.pipeline_component import PipelineComponent
from forte.process_manager import ProcessManager
from forte.utils import get_full_module_name

__all__ = [
    "BaseReader",
    "PackReader",
    'MultiPackReader'
]

logger = logging.getLogger(__name__)

process_manager = ProcessManager()


class BaseReader(PipelineComponent[PackType], ABC):
    """
        The basic data reader class.
        To be inherited by all data readers.
    """

    def __init__(self,
                 from_cache: bool = False,
                 cache_directory: Optional[str] = None,
                 append_to_cache: bool = False):
        """
        Args:
            from_cache (bool, optional): Decide whether to read from cache
                if cache file exists. By default (``False``), the reader will
                only read from the original file and use the cache file path
                for caching, it will not read from the cache_directory.
                If ``True``, the reader will try to read a datapack from the
                caching file.
            cache_directory (str, optional): The base directory to place the
                path of the caching files. Each collection is contained in one
                cached file, under this directory. The cached location for each
                collection is computed by :meth:`_cache_key_function`. Note:
                A collection is the data returned by :meth:`_collect`.
            append_to_cache (bool, optional): Decide whether to append write
                if cache file already exists.  By default (``False``), we
                will overwrite the existing caching file. If ``True``, we will
                cache the datapack append to end of the caching file.
        """
        self.from_cache = from_cache
        self._cache_directory = cache_directory
        self.component_name = get_full_module_name(self)
        self.append_to_cache = append_to_cache

    @staticmethod
    def default_configs():
        return {}

    @property
    def pack_type(self):
        raise NotImplementedError

    def __reader_name(self):
        return self.__class__.__name__

    # TODO: This should not be in the reader class.
    @staticmethod
    def serialize_instance(instance: PackType) -> str:
        """
        Serialize a pack to a string.
        """
        return instance.serialize()

    @staticmethod
    def deserialize_instance(string: str) -> PackType:
        """
        Deserialize an pack from a string.
        """
        return jsonpickle.decode(string)

    @abstractmethod
    def _collect(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """
        Gives an iterator of data objects
        each individual object should contain sufficient information
        needed to construct or locate a data pack in cache.
        For example: `data_source` can be a kwarg which is the path to a file
                     that a reader can take to read and parse a file.

        Args:
            args: Specify the data source.
            kwargs: Specify the data source.

        Returns: Iterator of collections that are sufficient to create one pack.

        """
        raise NotImplementedError

    def parse_pack(self, collection: Any) -> Iterator[PackType]:
        """
        Calls the _parse_pack method to create packs from the collection.
        This internally setup the component meta data. Users should implement
        the :meth:`_parse_pack` method.

        Args:
            collection:

        Returns:

        """
        process_manager.set_current_component(self.component_name)
        yield from self._parse_pack(collection)

    @abstractmethod
    def _parse_pack(self, collection: Any) -> Iterator[PackType]:
        """
        Gives an iterator of Packs parsed from a collection. Readers should
        implement this class to populate the class input.

        Args:
            collection: Object that can be parsed into a Pack

        Returns: Iterator[PackType]: Iterator of Packs

        """
        raise NotImplementedError

    @abstractmethod
    def _cache_key_function(self, collection: Any) -> str:
        """
        Computes the cache key based on the type of data.

        Args:
            collection: Any object that provides information
         to identify the name and location of the cache file

        Returns:
        """
        raise NotImplementedError

    # pylint: disable=unused-argument
    def text_replace_operation(self, text: str) -> ReplaceOperationsType:
        """
        Given the possibly noisy text, compute and return the
        replacement operations in the form of a list of (span, str)
        pairs, where the content in the span will be replaced by the
        corresponding str.

        Args:
            text: The original data text to be cleaned.

        Returns: List[Tuple[Tuple[int, int], str]]: the replacement operations

        """
        return []

    def _get_cache_location(self, collection: Any) -> Path:
        """
        Gets the path to the cache file for a collection

        Args:
            collection: information to compute cache key

        Returns: Path: file path to the cache file for a Pack

        """
        file_path = self._cache_key_function(collection)
        return Path(os.path.join(str(self._cache_directory), file_path))

    def _lazy_iter(self, *args, **kwargs):
        for collection in self._collect(*args, **kwargs):
            if self.from_cache:
                for pack in self.read_from_cache(
                        self._get_cache_location(collection)):
                    yield pack
            else:
                not_first = False
                for pack in self.parse_pack(collection):
                    # write to the cache if _cache_directory specified

                    if self._cache_directory is not None:
                        self.cache_data(
                            collection, pack, not_first)

                    if not isinstance(pack, self.pack_type):
                        raise ValueError(
                            f"No Pack object read from the given "
                            f"collection {collection}, returned {type(pack)}."
                        )
                    not_first = True
                    yield pack

    def iter(self, *args, **kwargs) -> Iterator[PackType]:
        """
         An iterator over the entire dataset, giving all Packs processed
         as list or Iterator depending on `lazy`, giving all the Packs read
         from the data source(s). If not reading from cache,
         should call collect()

        Args:
            args: One or more input data sources, for example, most
        DataPack readers accept `data_source` as file/folder path
            kwargs: Iterator of DataPacks.

        Returns:

        """
        yield from self._lazy_iter(*args, **kwargs)

    def cache_data(self,
                   collection: Any,
                   pack: PackType,
                   append: bool):
        """
        Specify the path to the cache directory.

        After you call this method, the dataset reader will use it's
        cache_directory to store a cache of :class:`BasePack` read
        from every document passed to :func:`read`, serialized as one
        string-formatted :class:`BasePack`. If the cache file for a given
        ``file_path`` exists, we read the :class:`BasePack` from the cache
        (using :func:`deserialize_instance`).  If the cache file does not
        exist, we will `create` it on our first pass through the data (using
        :func:`serialize_instance`).

        Args:
            collection: The collection is a piece of data from the
            _collect function, to be read to produce DataPack(s).
            During caching, a cache key is computed based on the data in this
            collection.
            pack: The data pack to be cached.
            append: Whether to allow appending to the cache.

        Returns:

        """
        if not self._cache_directory:
            raise ValueError(f"Can not cache without a cache_directory!")

        os.makedirs(self._cache_directory, exist_ok=True)

        cache_filename = os.path.join(
            self._cache_directory,
            self._get_cache_location(collection)
        )

        logger.info("Caching pack to %s", cache_filename)
        if append:
            with open(cache_filename, 'a') as cache:
                cache.write(self.serialize_instance(pack) + "\n")
        else:
            with open(cache_filename, 'w') as cache:
                cache.write(self.serialize_instance(pack) + "\n")

    def read_from_cache(self, cache_filename: Path) -> Iterator[PackType]:
        """
        Reads one or more Packs from a cache_filename,
        and yields Pack(s) from the cache file.

        Args:
            cache_filename: Path to the cache file

        Returns: List of cached data packs.

        """
        logger.info("reading from cache file %s", cache_filename)
        with cache_filename.open("r") as cache_file:
            for line in cache_file:
                pack = self.deserialize_instance(line.strip())
                if not isinstance(pack, self.pack_type):
                    raise TypeError(f"Pack deserialized from {cache_filename} "
                                    f"is {type(pack)},"
                                    f"but expect {self.pack_type}")
                yield pack

    def finish(self, resources: Resources):
        pass


class PackReader(BaseReader[DataPack], ABC):
    """
        A Pack Reader reads data into DataPacks.
    """

    @property
    def pack_type(self):
        return DataPack

    def set_text(self, pack: DataPack, text: str):
        """
        Assign the text value to the DataPack. This function will pass the
        text_replace_operation to the DataPack to conduct the pre-processing
        step.

        Args:
            pack: The datapack to assign value for.
            text: The original text to be recorded in this dataset

        Returns:

        """
        pack.set_text(text, replace_func=self.text_replace_operation)


class MultiPackReader(BaseReader[MultiPack], ABC):
    """
    The basic MultiPack data reader class.
    To be inherited by all data readers which return MultiPack.
    """

    @property
    def pack_type(self):
        return MultiPack
