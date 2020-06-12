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

from typing import Iterator

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.multi_pack import MultiPack
from forte.data.readers.base_reader import MultiPackReader
from ft.onto.base_ontology import Document


class EvalReader(MultiPackReader):

    # pylint: disable=no-self-use,unused-argument
    def _cache_key_function(self, collection) -> str:
        return "cached_string_file"

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, resources: Resources, configs: Config):
        self.resource = resources
        self.config = configs

    # pylint: disable=no-self-use
    def _collect(self, *args, **kwargs) -> Iterator[str]:
        file_path = args[0]
        with open(file_path, "r") as f:
            for line in f:
                yield line

    def _parse_pack(self, data_source: str) -> Iterator[MultiPack]:
        fields = data_source.split("\t")
        multi_pack = self.new_pack()

        data_pack = multi_pack.add_pack(self.config.pack_name)

        data_pack.pack_name = fields[0]
        data_pack.set_text(fields[1])

        Document(pack=data_pack, begin=0, end=len(fields[1]))

        yield multi_pack

    @classmethod
    def default_configs(cls):
        return {
            "pack_name": "query"
        }
