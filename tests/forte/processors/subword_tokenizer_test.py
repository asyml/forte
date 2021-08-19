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

import unittest
from typing import Dict

from ddt import data, ddt

from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.processors.misc import WhiteSpaceTokenizer
from forte.processors.nlp import SubwordTokenizer
from ft.onto.base_ontology import Subword


@ddt
class TestSubWordTokenizer(unittest.TestCase):
    @data(
        "GP contacted Harefield Hospital at Hillingdon in north London.",
        "Forte can run a subword tokenizer from Texar, by auto aligning.",
        "Handling unknown token like a chinese work: 铁.",
        "Macroglossum vidua is a moth of the family Sphingidae. It is known "
        "from north-eastern Papua New Guinea. The length of the forewings is "
        "about 22 mm. It is similar to Macroglossum glaucoptera, Macroglossum "
        "corythus luteata and Macroglossum sylvia, but recognisable by the "
        "dirty grey colour of the underside of the palpus, the greyish of the "
        "bases of the wing undersides and by the broad antemedian band of the "
        "forewing upperside. The head and thorax uppersides have no dark "
        "mesial stripe. The underside of the palpus and middle of the thorax "
        "are dirty grey, the white scaling mixed with drab-brown scales, "
        "the sides darker. The abdomen underside is grey. Both wing "
        "undersides are dark walnut-brown, dull, becoming somewhat olive "
        "distally, without a distinct brown border. The bases are faintly "
        "greyish. The hindwing upperside has an interrupted yellow "
        "band.\n\nReferences\n",
        "Balkanatolia 2-Annemden Rumeli Türküleri-Kalan-Turkey\n\nNotes and "
        "references\n\n\nExternal links\n\n* Rateyourmusic.com — Yıldız "
        "İbrahimova \n* Agency for Bulgarian Artists — photo and biography "
        "highlights in Bulgarian \n* International Famagusta Festival - "
        "Yıldız İbrahimova",
    )
    def test_tokenizer_auto(self, input_data):
        tokenizer = SubwordTokenizer()
        self.pl = (
            Pipeline[DataPack]()
            .set_reader(StringReader())
            .add(
                tokenizer, config={"tokenizer_configs": {"do_lower_case": True}}
            )
            .initialize()
        )

        # Take the vocabulary used by the tokenizer.
        self.vocab: Dict[str, str] = tokenizer.tokenizer.vocab
        for pack in self.pl.process_dataset(input_data):
            for subword in pack.get(Subword):
                if subword.is_unk:
                    assert subword.vocab_id == 100
                else:
                    subword_repr = (
                        subword.text
                        if subword.is_first_segment
                        else "##" + subword.text
                    )
                    if not (
                        subword_repr in self.vocab
                        or subword_repr.lower() in self.vocab
                    ):
                        assert False

    @data(
        "Balkanatolia 2-Annemden Rumeli Türküleri-Kalan-Turkey\n\nNotes and "
        "references\n\n\nExternal links\n\n* Rateyourmusic.com — Yıldız "
        "İbrahimova \n* Agency for Bulgarian Artists — photo and biography "
        "highlights in Bulgarian \n* International Famagusta Festival - "
        "Yıldız İbrahimova"
    )
    def test_tokenizer_unicode(self, input_data):
        self.pl = (
            Pipeline[DataPack]()
            .set_reader(StringReader())
            .add(WhiteSpaceTokenizer())
            .add(
                SubwordTokenizer(),
                config={
                    "tokenizer_configs": {"do_lower_case": True},
                    "token_source": "ft.onto.base_ontology.Token",
                },
            )
            .initialize()
        )

        for pack in self.pl.process_dataset(input_data):
            subwords = list(pack.get(Subword))
            self.assertEqual(len(subwords), 57)
            self.assertEqual(subwords[-1].text, "İbrahimova")
            self.assertTrue(subwords[-1].is_unk)
