# Copyright 2020 The Forte Authors. All Rights Reserved.
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
Unit tests for Random Word Splitting Data Augmentation processor
"""

import unittest
import random

from ddt import ddt, data, unpack
from forte.data.selector import AllPackSelector
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack
from forte.data.readers import StringReader
from forte.data.caster import MultiPackBoxer
from forte.processors.misc import PeriodSentenceSplitter
from forte.processors.data_augment.algorithms.word_splitting_processor import (
    RandomWordSplitDataAugmentProcessor,
)

from forte.processors.misc import WhiteSpaceTokenizer, EntityMentionInserter
from ft.onto.base_ontology import Token, EntityMention


@ddt
class TestWordSplittingProcessor(unittest.TestCase):
    def setUp(self):
        random.seed(8)
        self.nlp = Pipeline[MultiPack]()

        boxer_config = {"pack_name": "input_src"}
        entity_config = {"entities_to_insert": ["Mary", "station"]}
        self.nlp.set_reader(reader=StringReader())
        self.nlp.add(component=EntityMentionInserter(), config=entity_config)
        self.nlp.add(PeriodSentenceSplitter())
        self.nlp.add(component=MultiPackBoxer(), config=boxer_config)
        self.nlp.add(
            component=WhiteSpaceTokenizer(), selector=AllPackSelector()
        )

    @data(
        (
            [
                "Mary and Samantha arrived at the bus station on time . "
                "But they had to wait until noon for the bus ."
            ],
            [
                "Mary and Samantha arrived at the bus statio n on time . "
                "But t hey h ad to wait until noon for the bus ."
            ],
            [
                [
                    "Mary",
                    "and",
                    "Samantha",
                    "arrived",
                    "at",
                    "the",
                    "bus",
                    "statio",
                    " n",
                    "on",
                    "time",
                    ".",
                    "But",
                    "t",
                    " hey",
                    "h",
                    " ad",
                    "to",
                    "wait",
                    "until",
                    "noon",
                    "for",
                    "the",
                    "bus",
                    ".",
                ]
            ],
            [["station", "they", "had"]],
            [["Mary", "statio n", "t hey"]],
        )
    )
    @unpack
    def test_word_splitting_processor(
        self,
        texts,
        expected_outputs,
        expected_tokens,
        unnecessary_tokens,
        new_entities,
    ):
        self.nlp.add(component=RandomWordSplitDataAugmentProcessor())
        self.nlp.initialize()

        for idx, m_pack in enumerate(self.nlp.process_dataset(texts)):

            aug_pack = m_pack.get_pack("augmented_input_src")

            self.assertEqual(aug_pack.text, expected_outputs[idx])

            for j, token in enumerate(aug_pack.get(Token)):
                self.assertEqual(token.text, expected_tokens[idx][j])

            for token in unnecessary_tokens[idx]:
                self.assertNotIn(token, aug_pack.text)

            for j, token in enumerate(aug_pack.get(EntityMention)):
                self.assertEqual(token.text, new_entities[idx][j])


if __name__ == "__main__":
    unittest.main()
