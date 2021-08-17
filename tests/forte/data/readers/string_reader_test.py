# Copyright 2019 The Forte Authors. All Rights Reserved.
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
Unit test for StringReader.
"""

import os
import unittest
from pathlib import Path

from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline


class StringReaderPipelineTest(unittest.TestCase):
    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "forte/data/readers/tests/"

        self.pl1: Pipeline = Pipeline[DataPack]()
        self._cache_directory = Path(os.path.join(os.getcwd(), "cache_data"))
        self.pl1.set_reader(StringReader())
        self.pl1.initialize()

        self.pl2: Pipeline = Pipeline[DataPack]()
        self.pl2.set_reader(StringReader())
        self.pl2.initialize()

        self.text = (
            "The plain green Norway spruce is displayed in the gallery's "
            "foyer. Wentworth worked as an assistant to sculptor Henry Moore "
            "in the late 1960s. His reputation as a sculptor grew in the "
            "1980s."
        )

    def test_reader(self):
        self._process()
        self._read_caching()

    def _process(self):
        doc_exists = False
        for pack in self.pl1.process_dataset([self.text]):
            doc_exists = True
            self.assertEqual(self.text, pack.text)
        self.assertTrue(doc_exists)

    def _read_caching(self):
        doc_exists = False
        # get processed pack from dataset
        for pack in self.pl2.process_dataset([self.text]):
            doc_exists = True
            self.assertEqual(self.text, pack.text)
        self.assertTrue(doc_exists)

    def tearDown(self):
        os.system("rm -r {}".format(self._cache_directory))


if __name__ == "__main__":
    unittest.main()
