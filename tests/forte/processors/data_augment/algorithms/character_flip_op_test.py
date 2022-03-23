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
Unit tests for character flip op
"""

import random
import unittest
from forte.data.data_pack import DataPack
from ft.onto.base_ontology import Token
from forte.processors.data_augment.algorithms.character_flip_op import (
    CharacterFlipOp,
)


class TestCharacterFlipOp(unittest.TestCase):
    def setUp(self):
        self.test = CharacterFlipOp(
            configs={
                "prob": 0.3,
            }
        )

    def test_replace(self):
        random.seed(42)
        data_pack = DataPack()
        test_string = "The lazy fox jumped over the fence"
        test_result = "T/-/3 lazy f0>< jumpe|) oveI2 th3 fe^ce"
        data_pack.set_text(test_string)
        token_1 = Token(data_pack, 0, len(test_string))
        data_pack.add_entry(token_1)

        augmented_data_pack = self.test.perform_augmentation(data_pack)

        self.assertIn(augmented_data_pack.text, test_result)


if __name__ == "__main__":
    unittest.main()
