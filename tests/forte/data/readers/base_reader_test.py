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
Unit tests for BaseReader.
"""
import os
import unittest
from typing import List

from forte.data.data_pack import DataPack
from forte.data.readers.plaintext_reader import PlainTextReader
from forte.processors.base.pack_processor import PackProcessor
from forte.pipeline import Pipeline


class DummyPackProcessor(PackProcessor):
    def _process(self, input_pack: DataPack):
        pass


class BaseReaderTest(unittest.TestCase):
    def setUp(self):
        root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )

        # Define and config the Pipeline
        self.dataset_path = os.path.join(
            root_path, "data_samples/base_reader_test/"
        )

        self.nlp = Pipeline[DataPack]()

        self.reader = PlainTextReader(cache_in_memory=True)
        self.nlp.set_reader(self.reader)
        self.nlp.add(DummyPackProcessor())

        self.nlp.initialize()

    def test_memory_cache(self):
        parsed_packs: List[DataPack] = []

        for pack in self.nlp.process_dataset(self.dataset_path):
            parsed_packs.append(pack)

        self.assertTrue(self.reader._cache_ready)
        self.assertEqual(len(self.reader._data_packs), 3)
        self.assertEqual(parsed_packs, self.reader._data_packs)

        # Test reinitialize will clear memory cache
        self.nlp.initialize()

        self.assertFalse(self.reader._cache_ready)
        self.assertTrue(len(self.reader._data_packs) == 0)


if __name__ == "__main__":
    unittest.main()
