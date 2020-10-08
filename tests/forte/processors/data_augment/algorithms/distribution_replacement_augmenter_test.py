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
Unit tests for word replacement by distribution data augmenter.
"""

import unittest
import os

from forte.processors.data_augment.algorithms.distribution_replacement_augmenter import \
    DistributionReplacementAugmenter


class TestDictionaryReplacementAugmenter(unittest.TestCase):
    def setUp(self):
        self.augmenter = DistributionReplacementAugmenter(
            configs={"lang": "english",
                     "distribution": "unigram",
                     "unigram": os.path.join(os.path.dirname(__file__),
                                             "sample_unigram.txt")})

    def test_unigram_distribution(self):
        word = "computer"
        unigram = ['apple', 'banana', 'watermelon', 'lemon', 'orange']
        self.assertIn(self.augmenter.augment(word), unigram)


if __name__ == "__main__":
    unittest.main()
