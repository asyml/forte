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
Unit tests for dictionary word replacement data augmenter.
"""

import unittest

from forte.processors.data_augment.algorithms.back_translation_augmenter \
    import BackTranslationAugmenter


class TestBackTranslationAugmenter(unittest.TestCase):
    def setUp(self):
        self.bta = BackTranslationAugmenter(
            configs={
                "src_language": "en",
                "tgt_language": "de",
            }
        )

    def test_augmenter(self):
        print(self.bta.augment(
            "Mary and Samantha arrived at the bus station early but waited until noon for the bus."
        ))



if __name__ == "__main__":
    unittest.main()
