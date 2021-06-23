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
import re

from forte.data import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Sentence, Token

__all__ = [
    "PeriodSentenceSplitter",
    "WhiteSpaceTokenizer",
]


class PeriodSentenceSplitter(PackProcessor):
    """
    A processor that create sentences based on periods.
    """

    def _process(self, input_pack: DataPack):
        pattern = "\\.\\s*"
        start = 0

        for m in re.finditer(pattern, input_pack.text):
            end = m.end()
            Sentence(input_pack, start, end)
            start = end

        if start < len(input_pack.text):
            input_pack.add_entry(
                Sentence(input_pack, start, len(input_pack.text))
            )


class WhiteSpaceTokenizer(PackProcessor):
    """
    A simple processor that split the tokens based on white space.
    """

    def _process(self, input_pack: DataPack):
        pattern = r"\s+"
        start = 0

        for m in re.finditer(pattern, input_pack.text):
            input_pack.add_entry(Token(input_pack, start, m.start()))
            start = m.end()

        if start < len(input_pack.text):
            input_pack.add_entry(Token(input_pack, start, len(input_pack.text)))
