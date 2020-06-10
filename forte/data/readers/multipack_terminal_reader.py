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
# pylint: disable=attribute-defined-outside-init
"""
The reader that reads text data from a terminal and packs into a Multipack.
"""
import logging
from typing import Iterator, Dict, Any

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.multi_pack import MultiPack
from forte.data.readers.base_reader import MultiPackReader
from ft.onto.base_ontology import Utterance

logger = logging.getLogger(__name__)

__all__ = [
    "MultiPackTerminalReader"
]


class MultiPackTerminalReader(MultiPackReader):
    r"""A reader designed to read text from the terminal."""

    # pylint: disable=useless-super-delegation
    def __init__(self):
        super().__init__()

    # pylint: disable=unused-argument
    def initialize(self, resources: Resources, configs: Config):
        self.resource = resources
        self.config = configs

    # pylint: disable=unused-argument
    def _cache_key_function(self, collection) -> str:
        return "cached_string_file"

    def _collect(self) -> Iterator[str]:  # type: ignore
        # This allows the user to pass in either one single string or a list of
        # strings.
        while True:
            try:
                data = input("Enter your query here: ")
                if len(data) == 0:
                    continue
                yield data
            except EOFError:
                print()
                break

    def _parse_pack(self, data_source: str) -> Iterator[MultiPack]:
        r"""Takes a raw string and converts into a MultiPack.

        Args:
            data_source: str that contains text of a document.

        Returns: MultiPack containing a datapack for the current query.
        """
        multi_pack = self.new_pack()

        # use context to build the query
        if self.resource.get("user_utterance"):
            multi_pack.add_pack_(
                self.resource.get("user_utterance")[-1], "user_utterance")

        if self.resource.get("bot_utterance"):
            multi_pack.add_pack_(
                self.resource.get("bot_utterance")[-1], "bot_utterance")

        pack = multi_pack.add_pack(self.config.pack_name)
        pack.set_text(data_source, replace_func=self.text_replace_operation)

        Utterance(pack, 0, len(data_source))

        yield multi_pack

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        configs = super().default_configs()
        configs["pack_name"] = "query"
        return configs
