# Copyright 2021 The Forte Authors. All Rights Reserved.
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
The module contains assorted common readers.
"""
import logging
from typing import Iterator, Dict, Any

from forte.data import DataPack
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Utterance

logger = logging.getLogger(__name__)

__all__ = [
    "TerminalReader"
]


class TerminalReader(PackReader):
    r"""A reader designed to read text from the terminal."""

    # pylint: disable=unused-argument
    def _cache_key_function(self, collection) -> str:
        return "cached_string_file"

    def _collect(self) -> Iterator[str]:  # type: ignore
        # This allows the user to pass in either one single string or a list of
        # strings.
        while True:
            try:
                data = input(self.configs.prompt_text)
                if len(data) == 0:
                    continue
                yield data
            except EOFError:
                print()
                break

    def _parse_pack(self, data_source: str) -> Iterator[DataPack]:
        pack = DataPack()
        self.set_text(pack, data_source)
        Utterance(pack, 0, len(data_source))
        yield pack

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        Defines the default configuration for this class, available options are:

        .. code-block:: python

            {
                "prompt_text": "Enter your query here: ",
            }

        Here:

        `prompt_text` defines the text shown on the terminal as a prompt
        for the user.

        Returns: The default configuration values as a dict.
        """
        configs = super().default_configs()
        configs.update({
            "pack_name": "query",
            "prompt_text": "Enter your query here: ",
        })

        configs["pack_name"] = "query"
        return configs
