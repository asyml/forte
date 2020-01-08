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

from typing import Iterator, List

from texar.torch import HParams
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader


class DeserializeReader(PackReader):

    # pylint: disable=no-self-use,unused-argument
    def _cache_key_function(self, collection) -> str:
        return "cached_string_file"

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, resource: Resources, configs: HParams):
        self.resource = resource
        self.config = configs

    # pylint: disable=no-self-use
    def _collect(self, *args, **kwargs) -> Iterator[str]:
        data_packs: List[str] = args[0]
        
        yield from data_packs

    def _parse_pack(self, data_source: str) -> Iterator[DataPack]:
        yield self.deserialize_instance(data_source)
