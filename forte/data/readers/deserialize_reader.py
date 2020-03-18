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
from typing import Iterator, List

from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader


class BaseDeserializeReader(PackReader, ABC):
    # pylint: disable=unused-argument
    def _cache_key_function(self, collection) -> str:
        return "cached_string_file"

    def _parse_pack(self, data_source: str) -> Iterator[DataPack]:
        yield DataPack.deserialize(data_source)


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

    @staticmethod
    def default_configs():
        return {
            "suffix": ".json"
        }
