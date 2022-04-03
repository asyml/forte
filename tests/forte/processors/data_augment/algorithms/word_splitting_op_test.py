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
from ft.onto.base_ontology import Token, EntityMention
from forte.data.selector import AllPackSelector
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack, DataPack
from forte.data.readers import StringReader
from forte.data.caster import MultiPackBoxer
from forte.processors.misc import PeriodSentenceSplitter
from forte.processors.data_augment.base_op_processor import BaseOpProcessor
from forte.processors.base import PackProcessor
from forte.processors.misc import WhiteSpaceTokenizer


class EntityMentionInserter(PackProcessor):
    """
    A simple processor that inserts Entity Mentions into the data pack.
    The input required is the annotations that wish to be tagged as Entity
    Mentions. If the given annotations are not present in the given data pack,
    an exception is raised.
    """

    def _process(self, input_pack: DataPack):
        entity_text = self.configs.entities_to_insert

        input_text = input_pack.text
        if not all(bool(entity in input_text) for entity in entity_text):
            raise Exception(
                "Entities to be added are not valid for the input text."
            )
        for entity in entity_text:
            start = input_text.index(entity)
            end = start + len(entity)
            entity_mention = EntityMention(input_pack, start, end)
            input_pack.add_entry(entity_mention)

    @classmethod
    def default_configs(cls):
        return {"entities_to_insert": []}


@ddt
class TestWordSplittingProcessor(unittest.TestCase):
    """
    This class tests the correctness of word splitting data augmentation.
    It runs the following checks:
    - If expected augmentation = augmented text returned by function.
    - If return text containes all annotations as expected.
    - If annotations that were supposed to be augmented are not
    contained in the returned text
    - If the entities are retained after augmentation.
    """

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
        entity_config = {
            "data_aug_op": "forte.processors.data_augment.algorithms.word_splitting_op.RandomWordSplitDataAugmentOp",
            "data_aug_op_config": {
                "other_entry_policy": {
                    "ft.onto.base_ontology.EntityMention": "auto_align"
                }
            },
        }
        self.nlp.add(
            component=BaseOpProcessor(),
            config=entity_config,
        )
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
