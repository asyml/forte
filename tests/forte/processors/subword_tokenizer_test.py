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
"""
Unit tests for stave processor.
"""

import os
import unittest
from typing import Dict

from ddt import data, ddt

from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.processors.nlp import SubwordTokenizer
from ft.onto.base_ontology import Subword


@ddt
class TestSubWordTokenizer(unittest.TestCase):
    def setUp(self):
        tokenizer = SubwordTokenizer()
        self.pl = Pipeline[DataPack]().set_reader(
            StringReader()).add(tokenizer).initialize()

        # Take the vocabulary used by the tokenizer.
        self.vocab: Dict[str, str] = tokenizer.tokenizer.vocab

    @data(
        "GP contacted Harefield Hospital at Hillingdon in north London.",
        "Forte can run a subword tokenizer from Texar, by auto aligning.",
        "Handling unknown token like a chinese work: 铁."
    )
    def test_tokenizer(self, input_data):
        for pack in self.pl.process_dataset(input_data):
            for subword in pack.get(Subword):
                if subword.is_unk:
                    assert subword.vocab_id == 100
                else:
                    subword_repr = subword.text if subword.is_first_segment \
                        else "##" + subword.text
                    if not (subword_repr in self.vocab
                            or subword_repr.lower() in self.vocab):
                        assert False
