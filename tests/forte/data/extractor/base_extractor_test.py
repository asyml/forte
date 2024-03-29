#  Copyright 2020 The Forte Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import unittest
import pickle as pkl
from typing import Optional

from forte.data import DataPack
from forte.data.converter import Feature
from forte.data.ontology import Annotation
from ft.onto.base_ontology import Token
from forte.data.base_extractor import BaseExtractor


class TestExtractor(BaseExtractor):
    def extract(
        self, pack: DataPack, context: Optional[Annotation] = None
    ) -> Feature:
        pass


class BaseExtractorTest(unittest.TestCase):
    def test_base_extractor(self):
        config = {
            "vocab_method": "indexing",
            "need_pad": True,
        }

        extractor = TestExtractor()
        extractor.initialize(config=config)

        new_extractor = pkl.loads(pkl.dumps(extractor))

        # Check state and from state
        self.assertNotEqual(new_extractor.vocab, None)

        # Check vocab_method
        self.assertEqual(extractor.vocab_method, "indexing")

        # Check wrapped functions for vocabulary
        self.assertEqual(extractor.get_pad_value(), 0)


if __name__ == "__main__":
    unittest.main()
