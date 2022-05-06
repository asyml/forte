# Copyright 2022 The Forte Authors. All Rights Reserved.
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
Unit tests for dictionary word replacement op.
"""

import unittest
from forte.data.data_pack import DataPack
from ft.onto.base_ontology import Phrase
from forte.processors.data_augment.algorithms.abbreviation_replacement_op import (
    AbbreviationReplacementOp,
)


class TestAbbreviationReplacementOp(unittest.TestCase):
    def setUp(self):
        self.abre = AbbreviationReplacementOp(
            configs={
                "dict_path": "https://raw.githubusercontent.com/abbeyyyy/"
                "JsonFiles/main/abbreviate.json",
                "prob": 1.0,
            }
        )

    def test_replace(self):
        data_pack_1 = DataPack()
        text_1 = "I will see you later!"
        data_pack_1.set_text(text_1)
        phrase_1 = Phrase(data_pack_1, 7, len(text_1) - 1)
        data_pack_1.add_entry(phrase_1)

        augmented_data_pack_1 = self.abre.perform_augmentation(data_pack_1)
        augmented_phrase_1 = list(
            augmented_data_pack_1.get("ft.onto.base_ontology.Phrase")
        )[0]

        self.assertIn(
            augmented_phrase_1.text,
            ["syl8r", "cul83r", "cul8r"],
        )

        # Empty phrase
        data_pack_2 = DataPack()
        data_pack_2.set_text(text_1)
        phrase_2 = Phrase(data_pack_2, 0, 0)
        data_pack_2.add_entry(phrase_2)

        augmented_data_pack_2 = self.abre.perform_augmentation(data_pack_2)
        augmented_phrase_2 = list(
            augmented_data_pack_2.get("ft.onto.base_ontology.Phrase")
        )[0]

        self.assertIn(
            augmented_phrase_2.text,
            [""],
        )

        # no abbreviation exist
        data_pack_3 = DataPack()
        data_pack_3.set_text(text_1)
        phrase_3 = Phrase(data_pack_3, 2, 6)
        data_pack_3.add_entry(phrase_3)

        augmented_data_pack_3 = self.abre.perform_augmentation(data_pack_3)
        augmented_phrase_3 = list(
            augmented_data_pack_3.get("ft.onto.base_ontology.Phrase")
        )[0]

        self.assertIn(
            augmented_phrase_3.text,
            ["will"],
        )

if __name__ == "__main__":
    unittest.main()
