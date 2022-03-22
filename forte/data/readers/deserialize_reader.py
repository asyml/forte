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
import gzip
import logging
import os
from abc import ABC, abstractmethod
from typing import Iterator, List, Any, Optional

from smart_open import open

from forte.common import Resources
from forte.common.configuration import Config
from forte.common.exception import ProcessExecutionException
from forte.data.base_reader import PackReader, MultiPackReader
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack

__all__ = [
    "RawDataDeserializeReader",
    "RecursiveDirectoryDeserializeReader",
    "DirPackReader",
    "MultiPackDirectoryReader",
    "MultiPackDeserializerBase",
    "SinglePackReader",
]


class BaseDeserializeReader(PackReader, ABC):
    def _cache_key_function(self, _) -> str:
        return "cached_string_file"

    def _parse_pack(self, data_source: Any) -> Iterator[DataPack]:
        if data_source is None:
            raise ProcessExecutionException(
                "Data source is None, cannot deserialize."
            )

        pack: DataPack = DataPack.deserialize(
            data_source,
            serialize_method=self.configs.serialize_method,
            zip_pack=self.configs.zip_pack,
        )

        if pack is None:
            raise ProcessExecutionException(
                f"Cannot recover pack from the following data source: \n"
                f"{data_source}"
            )

        yield pack

    @classmethod
    def default_configs(cls):
        r"""This defines a basic configuration structure for reader.

        Here:

          - zip_pack (bool): whether to zip the data pack. The default value is
            False.

          - indent (int): None for not indented, if larger than 0, the JSON
            files will be written in the with the provided indention. The
            default value is None.

          - serialize_method: The method used to serialize the data. Current
            available options are "jsonpickle" and "pickle". Default is
            "jsonpickle".

        Returns:
            The default configuration of this writer.
        """
        return {
            "zip_pack": False,
            "indent": None,
            "serialize_method": "jsonpickle",
        }


class RawDataDeserializeReader(BaseDeserializeReader):
    """
    This reader assumes the data passed in are raw DataPack strings.
    """

    def _collect(self, data_list: List[str]) -> Iterator[str]:  # type: ignore
        yield from data_list

    def _parse_pack(self, data_source: str) -> Iterator[DataPack]:
        yield DataPack.from_string(data_source)  # type: ignore


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

        Returns:
            Iterator of the data pack string from the directory.
        """
        if not os.path.exists(data_dir):
            raise ValueError(
                f"The provided directory [{data_dir}] does not " f"exist."
            )
        for root, _, files in os.walk(data_dir):
            for file in files:
                if not self.configs.suffix or file.endswith(
                    self.configs.suffix
                ):
                    yield os.path.join(root, file)

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
            None
        """
        return {
            "suffix": ".json",
        }


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
        yield data_path


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

    def _parse_pack(self, multi_pack_source: Any) -> Iterator[MultiPack]:
        m_pack: MultiPack = self._parse_multi_pack(multi_pack_source)

        for pid in m_pack.pack_ids():
            data_pack = self._get_pack(pid)
            if data_pack is None:
                logging.warning(
                    "Cannot locate the data pack with pid %d "
                    "for multi pack %d",
                    pid,
                    m_pack.pack_id,
                )
                break
            # Only in deserialization we can do this.
            m_pack.packs.append(data_pack)
        else:
            # No multi pack will be yield if there are packs not located.
            yield m_pack

    def _parse_multi_pack(self, multi_pack_source: Any) -> MultiPack:
        return MultiPack.deserialize(
            multi_pack_source,
            self.configs.serialize_method,
            self.configs.zip_pack,
        )

    @abstractmethod
    def _get_multipack_content(
        self, *args: Any, **kwargs: Any
    ) -> Iterator[Any]:
        """
        Implementation of this method should be responsible for yielding
        the multi packs.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def _get_pack(self, pack_id: int) -> Optional[DataPack]:
        """
        Implementation of this method should be responsible for returning the
        actual data pack based on the pack id.

        Args:
            pack_id: representing the id of the data pack.

        Returns:
            The content of this data pack. You can either:
              - return the raw data pack string.
              - return the data pack as parsed DataPack object.

        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls):
        r"""This defines a basic configuration structure for writer.

        Here:
          - serialize_method: The method used to serialize the data. Current
              available options are "jsonpickle" and "pickle". Default is
              "jsonpickle".

        Returns: The default configuration of this writer.
        """
        return {
            "serialize_method": "jsonpickle",
        }


class MultiPackDirectoryReader(MultiPackDeserializerBase):
    """
    This reader implements one particular way of deserializing Multipack, which
    can be used to read the output written by
    :class:`~forte.processors.base.writers.PackNameMultiPackWriter`. It assumes
    the multipack are stored in a directory, and the data packs are stored in
    a directory too (they can be the same directory).
    """

    def __init__(self):
        super().__init__()
        self._open = open

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if self.configs.zip_pack:
            self._open = gzip.open

    def _get_multipack_content(self) -> Iterator[str]:  # type: ignore
        # pylint: disable=protected-access
        for mp_path in os.listdir(self.configs.multi_pack_dir):
            if mp_path.endswith(self.configs.suffix):
                yield os.path.join(self.configs.multi_pack_dir, mp_path)

    def _get_pack(self, pack_id: int) -> Optional[DataPack]:
        pack_path = os.path.join(
            self.configs.data_pack_dir, f"{pack_id}{self.configs.suffix}"
        )
        if os.path.exists(pack_path):
            return DataPack.deserialize(
                pack_path,
                serialize_method=self.configs.serialize_method,
                zip_pack=self.configs.zip_pack,
            )
        else:
            return None

    @classmethod
    def default_configs(cls):
        """
        Defines the default configuration for the multi pack reader.

        Here:
          - multi_pack_dir (str): the directory specifying the path storing the
              main multi pack content

          - data_pack_dir (str) : the directory specifying the path storing the
              data pack content.

          - suffix (str): the suffix of the data packs to be read.

          - serialize_method (str): The method used to serialize the data, this
              should be the same as how serialization is done. The current
              options are "jsonpickle" and "pickle". The default method
              is "jsonpickle".

          - zip_pack (bool): whether to zip the data pack. The default value is
              False.
        Returns:

        """
        return {
            "multi_pack_dir": None,
            "data_pack_dir": None,
            "suffix": ".json",
            "serialize_method": "jsonpickle",
            "zip_pack": False,
        }


# A short name for this class.
DirPackReader = RecursiveDirectoryDeserializeReader
