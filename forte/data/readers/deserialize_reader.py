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
import os
from abc import ABC, abstractmethod

from typing import Iterator, List, Any

from forte.common.exception import ProcessExecutionException
from forte.data.data_pack import DataPack
from forte.data.data_utils import deserialize
from forte.data.multi_pack import MultiPack
from forte.data.readers.base_reader import PackReader, MultiPackReader

__all__ = [
    'RawDataDeserializeReader',
    'RecursiveDirectoryDeserializeReader',
    'DirPackReader',
    'MultiPackDirectoryReader',
]


class BaseDeserializeReader(PackReader, ABC):
    # pylint: disable=unused-argument
    def _cache_key_function(self, collection) -> str:
        return "cached_string_file"

    def _parse_pack(self, data_source: str) -> Iterator[DataPack]:
        if data_source is None:
            raise ProcessExecutionException(
                "Data source is None, cannot deserialize.")

        # pack: DataPack = DataPack.deserialize(data_source)
        pack: DataPack = deserialize(data_source)

        if pack is None:
            raise ProcessExecutionException(
                f"Cannot recover pack from the following data source: \n"
                f"{data_source}")

        yield pack


class RawDataDeserializeReader(BaseDeserializeReader):
    """
    This reader assumes the data passed in are raw DataPack strings.
    """

    def _collect(self, data_list: List[str]) -> Iterator[str]:  # type: ignore
        yield from data_list


class RecursiveDirectoryDeserializeReader(BaseDeserializeReader):
    """
    This reader find all the files under the directory and read each one as
    a DataPack.
    """

    def _collect(self, data_dir: str) -> Iterator[str]:  # type: ignore
        """
        This function will collect the files of the given directory. If the
         'suffix' field in the config is set, it will only take files matching
         that suffix. See :func:`~forte.data.readers.RecursiveDirectory
         DeserializeReader.default_configs` for the default configs.

        Args:
            data_dir: The root directory to search for the data packs.

        Returns:

        """
        for root, _, files in os.walk(data_dir):
            for file in files:
                if not self.configs.suffix or file.endswith(
                        self.configs.suffix):
                    with open(os.path.join(root, file)) as f:
                        yield f.read()

    @classmethod
    def default_configs(cls):
        return {
            "suffix": ".json"
        }


class MultiPackDeserializerBase(MultiPackReader):
    """
    This is a base implementation of deserializing multipacks, extend this
    reader by implementing the following functions:
      - Implement :func:`_get_multipack_str` to return the raw string of the
         multi pack from the data source (read from :func:`_collect`)
      - Implement :func:`_get_pack_str` to get the string of a particular
         data pack given the pack id.

    The data source (such as dataset path) should be passed in from the
     configs during `initialize`, since the sources for multipack is likely to
     be complex.

    The base reader will then construct the multipack based on these
      information.
    """

    def _collect(self) -> Iterator[Any]:  # type: ignore
        """
        This collect actually do not need any data source, it directly reads
        the data from the configurations.

        Returns:

        """
        for s in self._get_multipack_content():
            yield s

    def _parse_pack(self, multi_pack_str: str) -> Iterator[MultiPack]:
        # pylint: disable=protected-access
        m_pack: MultiPack = deserialize(multi_pack_str)

        for pid in m_pack._pack_ref:
            pack: DataPack = deserialize(self._get_pack_content(pid))
            m_pack._packs.append(pack)
        yield m_pack

    @abstractmethod
    def _get_multipack_content(self) -> Iterator[str]:
        """
        Implementation of this method should be responsible for yielding
         the raw content of the multi packs.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def _get_pack_content(self, pack_id: int) -> str:
        """
        Implementation of this method should be responsible for returning the
          raw string of the data pack from the pack id.

        Args:
            pack_id: representing the id of the data pack.

        Returns:

        """
        raise NotImplementedError


class MultiPackDirectoryReader(MultiPackDeserializerBase):
    """
    This reader implements one particular way of deserializing Multipack, which
    can be used to read the output written by
    :class:`~forte.processors.base.writers.PackNameMultiPackWriter`. It assumes
    the multipack are stored in a directory, and the data packs are stored in
    a directory too (they can be the same directory).
    """

    def _get_multipack_content(self) -> Iterator[str]:
        # pylint: disable=protected-access
        for f in os.listdir(self.configs.multi_pack_dir):
            if f.endswith(self.configs.pack_suffix):
                with open(os.path.join(
                        self.configs.multi_pack_dir, f)) as m_data:
                    yield m_data.read()

    def _get_pack_content(self, pack_id: int) -> str:
        with open(os.path.join(
                self.configs.data_pack_dir, f'{pack_id}.json')) as pack_data:
            return pack_data.read()

    @classmethod
    def default_configs(cls):
        return {
            "multi_pack_dir": None,
            "data_pack_dir": None,
            "pack_suffix": '.json'
        }


# A short name for this class.
DirPackReader = RecursiveDirectoryDeserializeReader
