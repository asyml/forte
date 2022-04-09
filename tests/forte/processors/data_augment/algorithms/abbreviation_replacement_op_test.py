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
from forte.data.data_pack import DataPack
from ft.onto.base_ontology import Token
from forte.processors.data_augment.algorithms.abbreviation_replacement_op import (
    AbbreviationReplacementOp,
)


class TestAbbreviationReplacementOp(unittest.TestCase):
    def setUp(self):
        self.abre = AbbreviationReplacementOp(
            configs={
                "prob": 1.0,
            }
        )

    def test_replace(self):
        data_pack = DataPack()
        text = "see you later"
        data_pack.set_text(text)
        token = Token(data_pack, 0, len(text))
        data_pack.add_entry(token)

        augmented_data_pack = self.abre.perform_augmentation(data_pack)

        augmented_token = list(augmented_data_pack.get('ft.onto.base_ontology.Token'))[0]

        self.assertIn(
            augmented_token.text,
            ["syl8r", "cul83r", "cul8r"],
        )


if __name__ == "__main__":
    unittest.main()
