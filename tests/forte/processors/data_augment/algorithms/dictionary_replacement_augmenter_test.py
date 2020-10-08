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

from forte.processors.data_augment.algorithms.dictionary_replacement_augmenter \
    import DictionaryReplacementAugmenter

from ft.onto.base_ontology import Token
from forte.data.data_pack import DataPack


class TestDictionaryReplacementAugmenter(unittest.TestCase):
    def setUp(self):
        self.dra = DictionaryReplacementAugmenter(configs={"lang": "eng"})

    def test_segmenter(self):
        data_pack = DataPack()
        data_pack.set_text("eat phone")
        token_1 = Token(data_pack, 0, 3)
        token_2 = Token(data_pack, 4, 9)
        token_1.pos = "VB"
        token_2.pos = None
        data_pack.add_entry(token_1)
        data_pack.add_entry(token_2)

        self.assertIn(
            self.dra.augment(token_1),
            [
                'eat', 'feed', 'eat on', 'consume',
                'eat up', 'use up', 'deplete', 'exhaust',
                'run through', 'wipe out', 'corrode', 'rust'
            ]
        )
        self.assertIn(
            self.dra.augment(token_2),
            [
                'telephone', 'phone', 'telephone set',
                'speech sound', 'sound', 'earphone', 'earpiece',
                'headphone', 'call', 'telephone', 'call up', 'ring'
            ]
        )


if __name__ == "__main__":
    unittest.main()
