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
Unit tests for data store related operations.
"""

import logging
import unittest
from sortedcontainers import SortedList

from forte.data.data_store import DataStore

logging.basicConfig(level=logging.DEBUG)


class DataStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_store = DataStore()
        # attribute fields for Document and Sentence entries
        self.data_store._type_attributes = {
            "Document": ["document_class", "sentiment", "classifications"],
            "Sentence": [
                "speaker",
                "part_id",
                "sentiment",
                "classification",
                "classifications",
            ],
        }
        # The order is [Document, Sentence]. Initialize 2 entries in each list.
        # Document entries have tid 1234, 3456.
        # Sentence entries have tid 9999, 1234567.
        # The type id for Document is 0, Sentence is 1.
        self.data_store.elements = [
            SortedList(
                [
                    ["Document", 1234, 0, 5, None, "Postive", None],
                    [
                        "Document",
                        3456,
                        10,
                        25,
                        "Doc class A",
                        "Negative",
                        "Class B",
                    ],
                ],
                key=lambda x: (x[2], x[3]),
            ),
            SortedList(
                [
                    [
                        "Sentence",
                        9999,
                        6,
                        9,
                        "teacher",
                        1,
                        "Postive",
                        None,
                        None,
                    ],
                    [
                        "Sentence",
                        1234567,
                        55,
                        70,
                        None,
                        None,
                        "Negative",
                        "Class C",
                        "Class D",
                    ],
                ],
                key=lambda x: (x[2], x[3]),
            ),
        ]
        self.data_store.entry_dict = {
            1234: ["Document", 1234, 0, 5, None, "Postive", None],
            3456: [
                "Document",
                3456,
                10,
                25,
                "Doc class A",
                "Negative",
                "Class B",
            ],
            9999: ["Sentence", 9999, 6, 9, "teacher", 1, "Postive", None, None],
            1234567: [
                "Sentence",
                1234567,
                55,
                70,
                None,
                None,
                "Negative",
                "Class C",
                "Class D",
            ],
        }

    def test_add_annotation_raw(self):
        # test add Document entry
        self.data_store.add_annotation_raw(0, 1, 5)
        # test add Sentence entry
        self.data_store.add_annotation_raw(1, 5, 8)
        num_doc = len(self.data_store.elements[0])
        num_sent = len(self.data_store.elements[1])

        self.assertEqual(num_doc, 3)
        self.assertEqual(num_sent, 3)
        self.assertEqual(len(self.data_store.entry_dict), 6)

    def test_get_attr(self):
        speaker = self.data_store.get_attr(9999, "speaker")
        classifications = self.data_store.get_attr(3456, "classifications")

        self.assertEqual(speaker, "teacher")
        self.assertEqual(classifications, "Class B")

        # Entry with such tid does not exist
        with self.assertRaises(ValueError):
            for doc in self.data_store.get_attr(1111, "speaker"):
                print(doc)

        # Get attribute field that does not exist
        with self.assertRaises(ValueError):
            for doc in self.data_store.get_attr(9999, "class"):
                print(doc)

    def test_set_attr(self):
        # change attribute
        self.data_store.set_attr(9999, "speaker", "student")
        # set attribute with originally none value
        self.data_store.set_attr(1234, "document_class", "Class D")
        speaker = self.data_store.get_attr(9999, "speaker")
        doc_class = self.data_store.get_attr(1234, "document_class")

        self.assertEqual(speaker, "student")
        self.assertEqual(doc_class, "Class D")

        # Entry with such tid does not exist
        with self.assertRaises(ValueError):
            for doc in self.data_store.set_attr(1111, "speaker", "human"):
                print(doc)

        # Set attribute field that does not exist
        with self.assertRaises(ValueError):
            for doc in self.data_store.set_attr(9999, "speak", "human"):
                print(doc)

    def test_get_entry(self):
        sent = self.data_store.get_entry(1234567)
        self.assertEqual(
            sent,
            [
                "Sentence",
                1234567,
                55,
                70,
                None,
                None,
                "Negative",
                "Class C",
                "Class D",
            ],
        )

        # Entry with such tid does not exist
        with self.assertRaises(ValueError):
            for doc in self.data_store.get_entry(1111):
                print(doc)

    def test_delete_entry(self):
        # In test_add_annotation_raw(), we add 2 entries. So 6 in total.
        self.data_store.delete_entry(1234567)
        self.data_store.delete_entry(1234)
        self.data_store.delete_entry(9999)
        # After 3 deletion. 3 left. (2 documents and 1 sentence)
        num_doc = len(self.data_store.elements[0])
        num_sent = len(self.data_store.elements[1])

        self.assertEqual(len(self.data_store.entry_dict), 3)
        self.assertEqual(num_doc, 2)
        self.assertEqual(num_sent, 1)

        # Entry with such tid does not exist
        with self.assertRaises(ValueError):
            for doc in self.data_store.delete_entry(1111):
                print(doc)


if __name__ == "__main__":
    unittest.main()
