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
import logging
import os
from abc import ABC, abstractmethod
from typing import Iterator, List, Any, Union, Optional

from smart_open import open

from forte.common.exception import ProcessExecutionException
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.base_reader import PackReader, MultiPackReader

__all__ = [
    "RawDataDeserializeReader",
    "RecursiveDirectoryDeserializeReader",
    "DirPackReader",
    "MultiPackDirectoryReader",
    "MultiPackDeserializerBase",
    "SinglePackReader",
]


class BaseDeserializeReader(PackReader, ABC):
    # pylint: disable=unused-argument
    def _cache_key_function(self, collection) -> str:
        return "cached_string_file"

    def _parse_pack(self, data_source: str) -> Iterator[DataPack]:
        if data_source is None:
            raise ProcessExecutionException(
                "Data source is None, cannot deserialize."
            )

        pack: DataPack = DataPack.deserialize(data_source)

        if pack is None:
            raise ProcessExecutionException(
                f"Cannot recover pack from the following data source: \n"
                f"{data_source}"
            )

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
    a DataPack. If the `suffix` configuration is provided, then only files that
    end with this suffix will be read, otherwise all files will be read.
    Compressed data are supported through smart open.
    """

    def _collect(self, data_dir: str) -> Iterator[str]:  # type: ignore
        """
        This function will collect the files of the given directory. If the
         'suffix' field in the config is set, it will only take files matching
         that suffix. See :func:`~forte.data.readers.RecursiveDirectory
         DeserializeReader.default_configs` for the default configs.

        Args:
            data_dir: The root directory to search for the data packs.

        Returns: Iterator of the data pack string from the directory.
        """
        for root, _, files in os.walk(data_dir):
            for file in files:
                if not self.configs.suffix or file.endswith(
                    self.configs.suffix
                ):
                    with open(os.path.join(root, file)) as f:
                        yield f.read()

    @classmethod
    def default_configs(cls):
        """
        Store the configs for this reader.

         .. code-block:: python

            {
                "suffix": ".json",
            }

        Here, "suffix" is used to finds files matching the suffix. The default
        value is `.json`. If None, then all files will be read.

        Returns:

        """
        configs = super().default_configs()
        configs.update(
            {
                "suffix": ".json",
            }
        )
        return configs


class SinglePackReader(BaseDeserializeReader):
    """
    This reader reader one file of the given path as a DataPack. Compressed
    files are supported.
    """

    def _collect(self, data_path: str) -> Iterator[str]:  # type: ignore
        """
        This function will collect data path as a single file.

        Args:
            data_path: The file to the data pack.

        Returns:
            Iterator of only one item, the data pack string itself.
        """
        with open(data_path) as f:
            yield f.read()


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

    def _collect(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        """
        This collect actually do not need any data source, it directly reads
        the data from the configurations.

        Returns:

        """
        for s in self._get_multipack_content(*args, **kwargs):
            yield s

    def _parse_pack(self, multi_pack_str: str) -> Iterator[MultiPack]:
        m_pack: MultiPack = MultiPack.deserialize(multi_pack_str)

        for pid in m_pack.pack_ids():
            p_content = self._get_pack_content(pid)
            if p_content is None:
                logging.warning(
                    "Cannot locate the data pack with pid %d "
                    "for multi pack %d",
                    pid,
                    m_pack.pack_id,
                )
                break
            pack: DataPack
            if isinstance(p_content, str):
                pack = DataPack.deserialize(p_content)
            else:
                pack = p_content
            # Only in deserialization we can do this.
            m_pack.packs.append(pack)
        else:
            # No multi pack will be yield if there are packs not located.
            yield m_pack

    @abstractmethod
    def _get_multipack_content(
        self, *args: Any, **kwargs: Any
    ) -> Iterator[str]:
        """
        Implementation of this method should be responsible for yielding
         the raw content of the multi packs.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def _get_pack_content(self, pack_id: int) -> Union[None, str, DataPack]:
        """
        Implementation of this method should be responsible for returning the
          raw string of the data pack from the pack id.

        Args:
            pack_id: representing the id of the data pack.

        Returns:
            The content of this data pack. You can either:
              - return the raw data pack string.
              - return the data pack as parsed DataPack object.

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

    def _get_multipack_content(self) -> Iterator[str]:  # type: ignore
        # pylint: disable=protected-access
        for f in os.listdir(self.configs.multi_pack_dir):
            if f.endswith(self.configs.pack_suffix):
                with open(
                    os.path.join(self.configs.multi_pack_dir, f)
                ) as m_data:
                    yield m_data.read()

    def _get_pack_content(self, pack_id: int) -> Optional[str]:
        pack_path = os.path.join(self.configs.data_pack_dir, f"{pack_id}.json")
        if os.path.exists(pack_path):
            with open(pack_path) as pack_data:
                return pack_data.read()
        else:
            return None

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update(
            {
                "multi_pack_dir": None,
                "data_pack_dir": None,
                "pack_suffix": ".json",
            }
        )
        return config


# A short name for this class.
DirPackReader = RecursiveDirectoryDeserializeReader
