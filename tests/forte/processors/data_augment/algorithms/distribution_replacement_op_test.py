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
Unit tests for distribution word replacement op.
"""
import unittest

from ft.onto.base_ontology import Token
from forte.data.data_pack import DataPack
from forte.processors.data_augment.algorithms.distribution_replacement_op import (
    DistributionReplacementOp,
)


class TestDistributionReplacementOp(unittest.TestCase):
    def setUp(self):
        data_pack = DataPack()
        self.word = "eat"
        data_pack.set_text(self.word)
        self.token = Token(data_pack, 0, 3)
        data_pack.add_all_remaining_entries()

        self.word_list = ["apple", "banana", "orange"]
        self.word_dict = {
                            "apple": 1,
                            "banana": 2,
                            "mango": 3
                        }

    def test_replace(self):
        configs = {
            "prob": 1.0,
            "uniform_sampler_data": self.word_list,
            "sampler_type": "uniform"
        }
        replacement = DistributionReplacementOp(configs)
        word = replacement.replace(self.token)
        self.assertIn(word[1], self.word_list)

        configs = {
            "prob": 0,
            "uniform_sampler_data": self.word_list,
            "sampler_type": "uniform"
        }
        replacement = DistributionReplacementOp(configs)
        word = replacement.replace(self.token)
        self.assertEqual(word[1], self.word)

        configs = {
            "prob": 1.0,
            "sampler_type": "unigram",
            "unigram_sampler_data": self.word_dict
        }
        replacement = DistributionReplacementOp(configs)
        word = replacement.replace(self.token)
        self.assertIn(word[1], self.word_dict.keys())

if __name__ == "__main__":
    unittest.main()
