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
Tests for classification dataset reader.
"""
import os
import unittest
import csv
from typing import Dict
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Body, Title, Document
from forte.common import Resources, ProcessorConfigError
from forte.data.readers import ClassificationDatasetReader
from forte.data.data_pack import DataPack


class ClassificationDatasetReaderTest(unittest.TestCase):
    def setUp(self):
        self.sample_file1: str = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                *([os.path.pardir] * 4),
                "data_samples/amazon_review_polarity_csv/sample.csv"
            )
        )

        self.expected_content: Dict[int, str] = {}
        with open(self.sample_file1, encoding="utf-8") as f:
            data = csv.reader(f, delimiter=",", quoting=csv.QUOTE_ALL)
            for line_id, line in enumerate(data):
                class_id, title, description = line
                class_id = int(class_id)
                self.expected_content[line_id] = (class_id, title, description)
        self.class_names1 = ["negative", "positive"]
        self.index2class1 = dict(enumerate(self.class_names1))

        self.class_idx_to_name = {
            1: "negative",
            2: "positive",
        }

    def test_classification_dataset_reader(self):
        # test incompatible forte data field `ft.onto.base_ontology.Document`

        with self.assertRaises(ProcessorConfigError):
            self.pipeline = Pipeline()
            self.pipeline.set_reader(
                ClassificationDatasetReader(),
                config={
                    "index2class": self.index2class1,
                    "skip_k_starting_lines": 0,
                    "forte_data_fields": [
                        "label",
                        "ft.onto.base_ontology.Title",
                        "ft.onto.base_ontology.Document",
                    ],
                },
            )
            self.pipeline.initialize()
        # test wrong length of forte_data_fields
        with self.assertRaises(ProcessorConfigError):
            self.pipeline = Pipeline()
            self.pipeline.set_reader(
                ClassificationDatasetReader(),
                config={
                    "index2class": self.index2class1,
                    "skip_k_starting_lines": 0,
                    "forte_data_fields": [
                        "label",
                        "ft.onto.base_ontology.Body",
                    ],
                },
            )
            self.pipeline.initialize()
            # length check happens while processing data
            for data_pack in self.pipeline.process_dataset(self.sample_file1):
                continue
        self.pipeline = Pipeline()
        self.pipeline.set_reader(
            ClassificationDatasetReader(),
            config={
                "forte_data_fields": [
                    "label",
                    "ft.onto.base_ontology.Title",
                    "ft.onto.base_ontology.Body",
                ],
                "index2class": self.index2class1,
                "skip_k_starting_lines": 0,
            },
        )
        self.pipeline.initialize()
        for data_pack in self.pipeline.process_dataset(self.sample_file1):
            (
                expected_class_id,
                expected_title,
                expected_content,
            ) = self.expected_content[data_pack.pack_name]
            self.assertIsInstance(data_pack, DataPack)
            # Test Article
            doc_entries = list(data_pack.get(Document))
            # in our example, we have one body instance
            # stores one input ontology
            # and document instance stores all concatenated text
            self.assertTrue(len(doc_entries) == 1)
            article: Document = doc_entries[0]
            self.assertIsInstance(article, Document)
            self.assertEqual(
                article.text, expected_title + "\n" + expected_content
            )
            # Test Document Class
            doc_class = article.document_class
            self.assertTrue(len(doc_class) == 1)
            # print(class_idx_to_name, expected_class_id)
            self.assertEqual(
                doc_class[0], self.class_idx_to_name[expected_class_id]
            )
            # Test Title
            title_entries = list(data_pack.get(Title))
            self.assertTrue(len(title_entries) == 1)
            title: Title = title_entries[0]

            self.assertEqual(title.text, expected_title)
            # Test Description
            content: Body = list(data_pack.get(Body))[0]
            self.assertEqual(content.text, expected_content)


if __name__ == "__main__":
    unittest.main()
