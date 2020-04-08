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
from abc import ABC
from typing import Iterator, List, Any, Dict

from forte.common import Resources
from forte.common.configuration import Config
from forte.common.exception import ProcessExecutionException
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.readers.base_reader import PackReader, MultiPackReader

__all__ = [
    'RawDataDeserializeReader',
    'RecursiveDirectoryDeserializeReader',
    'DirPackReader',
    'MultiPackDiskReader',
]


class BaseDeserializeReader(PackReader, ABC):
    # pylint: disable=unused-argument
    def _cache_key_function(self, collection) -> str:
        return "cached_string_file"

    def _parse_pack(self, data_source: str) -> Iterator[DataPack]:
        if data_source is None:
            raise ProcessExecutionException(
                "Data source is None, cannot deserialize.")

        pack: DataPack = DataPack.deserialize(data_source)

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


class MultiPackDiskReader(MultiPackReader):
    """
    This reader implements one particular way of deserializing Multipack, which
    is corresponding to the
    :class:`~forte.processors.base.writers.MultiPackWriter`

    in this format, the DataPacks are serialized on the side, and the Multipack
    contains references to them. The reader here assemble these information
    together.
    """

    def __init__(self):
        super().__init__()
        self.__pack_paths: Dict[int, str] = {}

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.__get_pack_paths()

    def _collect(self) -> Iterator[str]:  # type: ignore
        """
        This collect actually do not need any data source, it directly read
        the data from the configurations.

        Returns:

        """
        multi_idx_path = os.path.join(self.configs.data_path, 'multi.idx')

        if not os.path.exists(multi_idx_path):
            raise FileNotFoundError(
                f"Cannot find file {multi_idx_path}, the multi pack "
                f"serialization format may be unknown.")

        with open(multi_idx_path) as multi_idx:
            for line in multi_idx:
                _, multi_path = line.strip().split()
                yield multi_path

    def _parse_pack(self, multi_pack_path: str) -> Iterator[MultiPack]:
        # pylint: disable=protected-access
        with open(os.path.join(
                self.configs.data_path, multi_pack_path)) as m_data:
            m_pack: MultiPack = MultiPack.deserialize(m_data.read())

            for pid in m_pack._pack_ref:
                sub_pack_path = self.__pack_paths[pid]
                with open(os.path.join(
                        self.configs.data_path, sub_pack_path)) as pack_data:
                    pack: DataPack = DataPack.deserialize(pack_data.read())
                    # Add a reference count to this pack, because the multipack
                    # needs it.
                    self._pack_manager.reference_pack(pack)

            self._remap_packs(m_pack)
            yield m_pack

    def _remap_packs(self, multi_pack: MultiPack):
        """Need to call this after reading the relevant data packs"""
        # pylint: disable=protected-access
        new_pack_refs: List[int] = []
        new_inverse_refs: Dict[int, int] = {}
        for pid in multi_pack._pack_ref:
            remapped_id = self._pack_manager.get_remapped_id(pid)
            new_pack_refs.append(remapped_id)
            new_inverse_refs[remapped_id] = len(new_pack_refs) - 1

        multi_pack._pack_ref = new_pack_refs
        multi_pack._inverse_pack_ref = new_inverse_refs

    def __get_pack_paths(self):
        pack_idx_path = os.path.join(self.configs.data_path, 'pack.idx')

        if not os.path.exists(pack_idx_path):
            raise FileNotFoundError(
                f"Cannot find file {pack_idx_path}, the multi pack "
                f"serialization format may be unknown.")

        # Reade data packs paths first.
        with open(pack_idx_path) as pack_idx:
            for line in pack_idx:
                pid, pack_path = line.strip().split()
                self.__pack_paths[int(pid)] = pack_path

    @classmethod
    def default_configs(cls):
        return {
            "data_path": None
        }

    def _cache_key_function(self, collection: Any) -> str:
        pass


# A short name for this class.
DirPackReader = RecursiveDirectoryDeserializeReader
