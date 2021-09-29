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
Tests for ag_news_reader.
"""
import os
import unittest
from typing import Dict

from forte.pipeline import Pipeline
from ftx.onto.ag_news import Description
from ft.onto.base_ontology import Document, Title

from forte.data.readers import AGNewsReader
from forte.data.data_pack import DataPack


class AGNewsReaderTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()

        self.pipeline.set_reader(AGNewsReader())
        self.pipeline.initialize()

        self.sample_file: str = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                *([os.path.pardir] * 4),
                "data_samples/ag_news/sample.csv"
            )
        )

        self.expected_content: Dict[int, str] = {}
        with open(self.sample_file, "r") as file:
            for line_id, line in enumerate(file):
                data = line.strip().split(",")
                class_id, title, description = (
                    int(data[0].replace('"', "")),
                    data[1],
                    data[2],
                )
                self.expected_content[line_id] = (class_id, title, description)

        self.class_idx_to_name = {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Sci/Tech",
        }

    def test_ag_news_reader(self):
        for data_pack in self.pipeline.process_dataset(self.sample_file):
            (
                expected_class_id,
                expected_title,
                expected_desc,
            ) = self.expected_content[data_pack.pack_name]
            self.assertIsInstance(data_pack, DataPack)
            # Test Article
            doc_entries = list(data_pack.get(Document))
            self.assertTrue(len(doc_entries) == 1)
            article: Document = doc_entries[0]
            self.assertIsInstance(article, Document)
            self.assertEqual(
                article.text, expected_title + "\n" + expected_desc
            )
            # Test Document Class
            doc_class = article.document_class
            self.assertTrue(len(doc_class) == 1)
            self.assertEqual(
                doc_class[0], self.class_idx_to_name[expected_class_id]
            )
            # Test Title
            title_entries = list(data_pack.get(Title))
            self.assertTrue(len(title_entries) == 1)
            title: Title = title_entries[0]
            self.assertEqual(title.text, expected_title)
            # Test Description
            desc_entries = list(data_pack.get(Description))
            self.assertTrue(len(desc_entries) == 1)
            description: Description = desc_entries[0]
            self.assertEqual(description.text, expected_desc)


if __name__ == "__main__":
    unittest.main()
