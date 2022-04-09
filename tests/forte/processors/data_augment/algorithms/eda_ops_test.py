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
Unit tests for EDA data augment processors
"""

import unittest
import random

from forte.data.selector import AllPackSelector
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack
from forte.data.readers import StringReader
from forte.data.caster import MultiPackBoxer
from forte.processors.data_augment.base_op_processor import BaseOpProcessor
from forte.processors.misc import WhiteSpaceTokenizer
from ft.onto.base_ontology import Token

from ddt import ddt, data, unpack


@ddt
class TestEDADataAugmentProcessor(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.nlp = Pipeline[MultiPack]()

        boxer_config = {"pack_name": "input_src"}

        self.nlp.set_reader(reader=StringReader())
        self.nlp.add(component=MultiPackBoxer(), config=boxer_config)
        self.nlp.add(
            component=WhiteSpaceTokenizer(), selector=AllPackSelector()
        )

    @data(
        (
            [
                "Mary and Samantha arrived at the bus station "
                "early but waited until noon for the bus ."
            ],
            [
                "Mary early Samantha arrived at the bus station "
                "and but waited until for noon the bus ."
            ],
            [
                [
                    "Mary",
                    "early",
                    "Samantha",
                    "arrived",
                    "at",
                    "the",
                    "bus",
                    "station",
                    "and",
                    "but",
                    "waited",
                    "until",
                    "for",
                    "noon",
                    "the",
                    "bus",
                    ".",
                ]
            ],
        )
    )
    @unpack
    def test_random_swap(self, texts, expected_outputs, expected_tokens):

        swap_config = {
            "data_aug_op": "forte.processors.data_augment.algorithms.eda_ops.RandomSwapDataAugmentOp"
        }

        self.nlp.add(component=BaseOpProcessor(), config=swap_config)
        self.nlp.initialize()

        for idx, m_pack in enumerate(self.nlp.process_dataset(texts)):
            aug_pack = m_pack.get_pack("augmented_input_src")

            self.assertEqual(aug_pack.text, expected_outputs[idx])

            for j, token in enumerate(aug_pack.get(Token)):
                self.assertEqual(token.text, expected_tokens[idx][j])

    @data(
        (
            [
                "Mary and Samantha arrived at the bus station early "
                "but waited until noon for the bus ."
            ],
            [
                "await Mary and Samantha arrived at the bus station "
                "early but waited until noon for the bus ."
            ],
            [
                [
                    "await ",
                    "Mary",
                    "and",
                    "Samantha",
                    "arrived",
                    "at",
                    "the",
                    "bus",
                    "station",
                    "early",
                    "but",
                    "waited",
                    "until",
                    "noon",
                    "for",
                    "the",
                    "bus",
                    ".",
                ]
            ],
        )
    )
    @unpack
    def test_random_insert(self, texts, expected_outputs, expected_tokens):

        insert_config = {
            "data_aug_op": "forte.processors.data_augment.algorithms.eda_ops.RandomInsertionDataAugmentOp"
        }

        self.nlp.add(component=BaseOpProcessor(), config=insert_config)
        self.nlp.initialize()

        for idx, m_pack in enumerate(self.nlp.process_dataset(texts)):
            aug_pack = m_pack.get_pack("augmented_input_src")

            self.assertEqual(aug_pack.text, expected_outputs[idx])

            for j, token in enumerate(aug_pack.get(Token)):
                self.assertEqual(token.text, expected_tokens[idx][j])

    @data(
        (
            [
                "Mary and Samantha arrived at the bus station "
                "early but waited until noon for the bus ."
            ],
            ["Mary and   at  bus   but waited until  for the  ."],
            [
                [
                    "Mary",
                    "and",
                    "at",
                    "bus",
                    "but",
                    "waited",
                    "until",
                    "for",
                    "the",
                    ".",
                ]
            ],
        )
    )
    @unpack
    def test_random_delete(self, texts, expected_outputs, expected_tokens):

        insert_config = {
            "data_aug_op": "forte.processors.data_augment.algorithms.eda_ops.RandomDeletionDataAugmentOp",
            "data_aug_op_config": {"alpha": 0.5},
        }

        self.nlp.add(
            component=BaseOpProcessor(),
            config=insert_config,
        )
        self.nlp.initialize()

        for idx, m_pack in enumerate(self.nlp.process_dataset(texts)):
            aug_pack = m_pack.get_pack("augmented_input_src")

            self.assertEqual(aug_pack.text, expected_outputs[idx])

            for j, token in enumerate(aug_pack.get(Token)):
                self.assertEqual(token.text, expected_tokens[idx][j])


if __name__ == "__main__":
    unittest.main()
