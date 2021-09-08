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
from forte.processors.data_augment.algorithms.typo_replacement_op import (
    TypoReplacementOp,
)


class TestTypoReplacementOp(unittest.TestCase):
    def setUp(self):
        self.tyre = TypoReplacementOp(
            configs={
                "prob": 1.0,
                "typo_generator": "uniform",
            }
        )

    def test_replace(self):
        data_pack = DataPack()
        data_pack.set_text("auxiliary colleague apple")
        token_1 = Token(data_pack, 0, 9)
        token_2 = Token(data_pack, 10, 19)
        token_3 = Token(data_pack, 20, 25)
        data_pack.add_entry(token_1)
        data_pack.add_entry(token_2)
        data_pack.add_entry(token_3)

        self.assertIn(
            self.tyre.replace(token_1)[1],
            ["auxilliary", "auxilary", "auxillary"],
        )
        self.assertIn(self.tyre.replace(token_2)[1], ["collegue", "colleaque"])
        self.assertIn(self.tyre.replace(token_3)[1], ["apple"])


if __name__ == "__main__":
    unittest.main()
