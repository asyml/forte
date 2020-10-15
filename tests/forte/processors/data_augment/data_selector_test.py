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
Unit tests for data selector processor.
"""

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.base import LengthSelectorProcessor
from typing import List
import unittest


class TestDataSelectorProcessor(unittest.TestCase):
    def setUp(self):
        self.selector = LengthSelectorProcessor()
        self.selector.initialize(resources=None, configs={"max_length": 20})
        # Todo: configs doesn't update, still use default val?

    def test_length_selector(self):

        self.data:List[str] = ["Mary and Samantha arrived at the bus station early "
                          "but waited until noon for the bus.",
            "apple banana pear"]
        nlp = Pipeline[DataPack]()
        nlp.set_reader(reader=StringReader())
        nlp.add(self.selector)
        nlp.initialize()

        n_pack = 0
        for _, m_pack in enumerate(nlp.process_dataset(self.data)):
            self.assertEqual(m_pack.text, "apple banana pear")
            n_pack+=1
        self.assertEqual(n_pack, 1)


if __name__ == "__main__":
    unittest.main()
