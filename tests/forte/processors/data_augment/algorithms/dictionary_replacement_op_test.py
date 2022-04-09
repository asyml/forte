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
Unit tests for dictionary word replacement op.
"""

import unittest
from forte.processors.data_augment.algorithms.dictionary_replacement_op import (
    DictionaryReplacementOp,
)

from ft.onto.base_ontology import Token
from forte.data.data_pack import DataPack


class TestDictionaryReplacementOp(unittest.TestCase):
    def setUp(self):
        dict_name = (
            "forte.processors.data_augment."
            "algorithms.dictionary.WordnetDictionary"
        )
        self.dra = DictionaryReplacementOp(
            configs={
                "dictionary_class": dict_name,
                "prob": 1.0,
                "lang": "eng",
            }
        )

    def test_segmenter(self):
        data_pack = DataPack()
        data_pack.set_text("eat phone")
        token_1 = Token(data_pack, 0, 3)
        token_2 = Token(data_pack, 4, 9)
        token_1.pos = "VB"
        token_2.pos = None
        data_pack.add_entry(token_1)
        data_pack.add_entry(token_2)

        augmented_data_pack = self.dra.perform_augmentation(data_pack)

        augmented_tokens = list(
            augmented_data_pack.get("ft.onto.base_ontology.Token")
        )
        aug_token_1 = augmented_tokens[0]
        aug_token_2 = augmented_tokens[1]

        self.assertIn(
            aug_token_1.text,
            [
                "eat",
                "feed",
                "eat on",
                "consume",
                "eat up",
                "use up",
                "deplete",
                "exhaust",
                "run through",
                "wipe out",
                "corrode",
                "rust",
            ],
        )
        self.assertIn(
            aug_token_2.text,
            [
                "telephone",
                "phone",
                "telephone set",
                "speech sound",
                "sound",
                "earphone",
                "earpiece",
                "headphone",
                "call",
                "telephone",
                "call up",
                "ring",
            ],
        )


if __name__ == "__main__":
    unittest.main()
