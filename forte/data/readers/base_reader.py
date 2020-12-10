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
from typing import Any, Iterator, Optional, Union

from forte.common.exception import ProcessExecutionException
from forte.common.resources import Resources
from forte.data import data_utils
from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.types import ReplaceOperationsType
from forte.pipeline_component import PipelineComponent
from forte.utils.utils import get_full_module_name

__all__ = [
    "BaseReader",
    "PackReader",
    'MultiPackReader',
]

logger = logging.getLogger(__name__)


class BaseReader(PipelineComponent[PackType], ABC):
    r"""The basic data reader class. To be inherited by all data readers.

    Args:
        from_cache (bool, optional): Decide whether to read from cache
            if cache file exists. By default (``False``), the reader will
            only read from the original file and use the cache file path
            for caching, it will not read from the ``cache_directory``.
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

    def __init__(self,
                 from_cache: bool = False,
                 cache_directory: Optional[str] = None,
                 append_to_cache: bool = False,
                 cache_in_memory: bool = False):
        super().__init__()
        self.from_cache = from_cache
        self._cache_directory = cache_directory
        self.component_name = get_full_module_name(self)
        self.append_to_cache = append_to_cache
        self._cache_in_memory = cache_in_memory
        self._cache_ready: bool = False
        self._data_packs = []

    @classmethod
    def default_configs(cls):
        r"""Returns a `dict` of configurations of the reader with default
        values. Used to replace the missing values of input `configs`
        during pipeline construction.

        .. code-block:: python

            {
                "name": "reader"
            }
        """
        return {
            'name': 'reader'
        }

    @property
    def pack_type(self):
        raise NotImplementedError

    @abstractmethod
    def _collect(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        r"""Returns an iterator of data objects, and each individual object
        should contain sufficient information needed to construct or locate
        a data pack in cache.

        For example: `data_source` can be a ``kwarg`` which is the path to a
        file that a reader can take to read and parse a file.

        Args:
            args: Specify the data source.
            kwargs: Specify the data source.

        Returns: Iterator of collections that are sufficient to create one pack.
        """
        raise NotImplementedError

    def parse_pack(self, collection: Any) -> Iterator[PackType]:
        r"""Calls :meth:`_parse_pack` to create packs from the collection.
        This internally setup the component meta data. Users should implement
        the :meth:`_parse_pack` method.
        """
        if collection is None:
            raise ProcessExecutionException(
                "Got None collection, cannot parse as data pack.")

        for p in self._parse_pack(collection):
            p.add_all_remaining_entries(self.name)
            yield p

    @abstractmethod
    def _parse_pack(self, collection: Any) -> Iterator[PackType]:
        r"""Returns an iterator of Packs parsed from a collection. Readers
        should implement this class to populate the class input.

        Args:
            collection: Object that can be parsed into a Pack.

        Returns: Iterator of Packs.
        """
        raise NotImplementedError

    def _cache_key_function(self, collection: Any) -> Optional[str]:
        # pylint: disable=unused-argument
        r"""Computes the cache key based on the type of data.

        Args:
            collection: Any object that provides information to identify the
                name and location of the cache file
        """
        return None

    # pylint: disable=unused-argument
    def text_replace_operation(self, text: str) -> ReplaceOperationsType:
        r"""Given the possibly noisy text, compute and return the
        replacement operations in the form of a list of (span, str)
        pairs, where the content in the span will be replaced by the
        corresponding str.

        Args:
            text: The original data text to be cleaned.

        Returns (List[Tuple[Tuple[int, int], str]]): the replacement operations.
        """
        return []

    def _get_cache_location(self, collection: Any) -> str:
        r"""Gets the path to the cache file for a collection.

        Args:
            collection: information to compute cache key.

        Returns (Path): file path to the cache file for a Pack.
        """
        # pylint: disable=assignment-from-none
        file_path = self._cache_key_function(collection)
        if file_path is None:
            raise ProcessExecutionException(
                "Cache key is None. You probably set `from_cache` to true but "
                "fail to implement the _cache_key_function")

        return os.path.join(str(self._cache_directory), file_path)

    def _lazy_iter(self, *args, **kwargs):
        for collection in self._collect(*args, **kwargs):
            if self.from_cache:
                for pack in self.read_from_cache(
                        self._get_cache_location(collection)):
                    pack.add_all_remaining_entries()
                    yield pack
            else:
                not_first = False
                for pack in self.parse_pack(collection):
                    # write to the cache if _cache_directory specified
                    if self._cache_directory is not None:
                        self.cache_data(collection, pack, not_first)

                    if not isinstance(pack, self.pack_type):
                        raise ValueError(
                            f"No Pack object read from the given "
                            f"collection {collection}, returned {type(pack)}."
                        )

                    not_first = True
                    pack.add_all_remaining_entries()
                    yield pack

    def iter(self, *args, **kwargs) -> Iterator[PackType]:
        r"""An iterator over the entire dataset, giving all Packs processed
        as list or Iterator depending on `lazy`, giving all the Packs read
        from the data source(s). If not reading from cache, should call
        :meth:`collect`.

        Args:
            args: One or more input data sources, for example, most
                DataPack readers accept `data_source` as file/folder path.
            kwargs: Iterator of DataPacks.
        """
        if self._cache_in_memory and self._cache_ready:
            # Read from memory
            for pack in self._data_packs:
                yield pack
        else:
            # Read via parsing dataset
            for pack in self._lazy_iter(*args, **kwargs):
                if self._cache_in_memory:
                    self._data_packs.append(pack)
                yield pack

        self._cache_ready = True

    def cache_data(self, collection: Any, pack: PackType, append: bool):
        r"""Specify the path to the cache directory.

        After you call this method, the dataset reader will use its
        ``cache_directory`` to store a cache of :class:`BasePack` read
        from every document passed to :func:`read`, serialized as one
        string-formatted :class:`BasePack`. If the cache file for a given
        ``file_path`` exists, we read the :class:`BasePack` from the cache.
        If the cache file does not exist, we will `create` it on our first
        pass through the data.

        Args:
            collection: The collection is a piece of data from the
                :meth:`_collect` function, to be read to produce DataPack(s).
                During caching, a cache key is computed based on the data in
                this collection.
            pack: The data pack to be cached.
            append: Whether to allow appending to the cache.
        """
        if not self._cache_directory:
            raise ValueError("Can not cache without a cache_directory!")

        os.makedirs(self._cache_directory, exist_ok=True)

        cache_filename = os.path.join(
            self._cache_directory,
            self._get_cache_location(collection)
        )

        logger.info("Caching pack to %s", cache_filename)
        if append:
            with open(cache_filename, 'a') as cache:
                cache.write(pack.serialize() + "\n")
        else:
            with open(cache_filename, 'w') as cache:
                cache.write(pack.serialize() + "\n")

    def read_from_cache(
            self, cache_filename: Union[Path, str]) -> Iterator[PackType]:
        r"""Reads one or more Packs from ``cache_filename``, and yields Pack(s)
        from the cache file.

        Args:
            cache_filename: Path to the cache file.

        Returns: List of cached data packs.
        """
        logger.info("reading from cache file %s", cache_filename)
        with open(cache_filename, "r") as cache_file:
            for line in cache_file:
                pack = data_utils.deserialize(line.strip())
                if not isinstance(pack, self.pack_type):
                    raise TypeError(
                        f"Pack deserialized from {cache_filename} "
                        f"is {type(pack)}, but expect {self.pack_type}")
                yield pack

    def finish(self, resources: Resources):
        pass


class PackReader(BaseReader[DataPack], ABC):
    r"""A Pack Reader reads data into :class:`DataPack`.
    """

    @property
    def pack_type(self):
        return DataPack

    def set_text(self, pack: DataPack, text: str):
        r"""Assign the text value to the :class:`DataPack`. This function will
        pass the ``text_replace_operation`` to the :class:`DataPack` to conduct
        the pre-processing step.

        Args:
            pack: The :class:`DataPack` to assign value for.
            text: The original text to be recorded in this dataset.
        """
        pack.set_text(text, replace_func=self.text_replace_operation)


class MultiPackReader(BaseReader[MultiPack], ABC):
    r"""The basic :class:`MultiPack` data reader class. To be inherited by all
    data readers which return :class:`MultiPack`.
    """

    @property
    def pack_type(self):
        return MultiPack
