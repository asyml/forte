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
Tests for corpus_reader.
"""
import os
import unittest
from typing import Dict

from ft.onto.base_ontology import Document

from forte.data.readers import MSMarcoPassageReader
from forte.data.data_pack import DataPack


class MSMarcoPassageReaderTest(unittest.TestCase):

    def setUp(self):
        self.reader = MSMarcoPassageReader()

        self.data_dir = 'data_samples/ms_marco_passage_retrieval'

        corpus_file = os.path.join(self.data_dir, 'collection.tsv')
        self.expected_content = {}
        with open(corpus_file, 'r') as f:
            for line in f.readlines():
                key, value = tuple(line.split('\t', 1))
                self.expected_content[key] = value

    def test_ms_marco_passage_reader(self):
        actual_content: Dict[str, str] = {}
        for data_pack in self.reader.iter(self.data_dir):
            self.assertIsInstance(data_pack, DataPack)
            doc_entries = list(data_pack.get_entries_by_type(Document))
            self.assertTrue(len(doc_entries) == 1)
            doc_entry: Document = doc_entries[0]
            self.assertIsInstance(doc_entry, Document)
            actual_content[data_pack.meta.doc_id] = doc_entry.text

        self.assertDictEqual(actual_content, self.expected_content)


if __name__ == "__main__":
    unittest.main()
