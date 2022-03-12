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
import tempfile
import os

from forte.data.data_store import DataStore

logging.basicConfig(level=logging.DEBUG)


class DataStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_store = DataStore()
        # attribute fields for Document and Sentence entries
        self.data_store._type_attributes = {
            "ft.onto.base_ontology.Document": {
                "sentiment": 4,
                "classifications": 5,
            },
            "ft.onto.base_ontology.Sentence": {
                "sentiment": 4,
                "speaker": 5,
                "part_id": 6,
                "classification_test": 7,
                "classifications": 8,
                "temp": 9,
            },
        }
        # The order is [Document, Sentence]. Initialize 2 entries in each list.
        # Document entries have tid 1234, 3456.
        # Sentence entries have tid 9999, 1234567.
        # The type id for Document is 0, Sentence is 1.

        self.data_store._DataStore__elements = {
            "ft.onto.base_ontology.Document": SortedList(
                [
                    [
                        0,
                        5,
                        1234,
                        0,
                        "Postive",
                        None,
                        # None,
                    ],
                    [
                        10,
                        25,
                        3456,
                        0,
                        "Negative",
                        "Class B",
                        # "Doc class A",
                    ],
                ],
            ),
            "ft.onto.base_ontology.Sentence": SortedList(
                [
                    [
                        55,
                        70,
                        1234567,
                        1,
                        "Negative",
                        None,
                        None,
                        "Class C",
                        "Class D",
                        "abc",
                    ],
                    [6, 9, 9999, 1, "Postive", "teacher", 1, None, None, "cba"],
                ],
            ),
        }

        DataStore._type_attributes = {
            "ft.onto.base_ontology.Document": {
                "document_class": 4,
                "sentiment": 5,
                "classifications": 6,
            },
            "ft.onto.base_ontology.Sentence": {
                "speaker": 4,
                "part_id": 5,
                "sentiment": 6,
                "classification": 7,
                "classifications": 8,
            },
        }

    def test_pickle(self):
        with tempfile.TemporaryDirectory() as tempdir:
            tmpfilepath = os.path.join(tempdir, "temp.txt")
            self.data_store.serialize(tmpfilepath)
            temp = DataStore.deserialize(tmpfilepath)
            self.assertEqual(
                temp._DataStore__elements,
                {
                    "ft.onto.base_ontology.Document": SortedList(
                        [
                            [
                                10,
                                25,
                                3456,
                                0,
                                # "Doc class A",
                                None,
                                "Negative",
                                "Class B",
                            ],
                            [
                                0,
                                5,
                                1234,
                                0,
                                None,
                                "Postive",
                                None,
                            ],
                        ],
                    ),
                    "ft.onto.base_ontology.Sentence": SortedList(
                        [
                            [
                                55,
                                70,
                                1234567,
                                1,
                                None,
                                None,
                                "Negative",
                                None,
                                "Class D",
                            ],
                            [
                                6,
                                9,
                                9999,
                                1,
                                "teacher",
                                1,
                                "Postive",
                                None,
                                None,
                            ],
                        ],
                    ),
                },
            )

            temp = DataStore.deserialize(tmpfilepath, check_attribute=False)
            self.assertEqual(
                temp._DataStore__elements,
                {
                    "ft.onto.base_ontology.Document": SortedList(
                        [
                            [
                                0,
                                5,
                                1234,
                                0,
                                "Postive",
                                None,
                                # None,
                            ],
                            [
                                10,
                                25,
                                3456,
                                0,
                                "Negative",
                                "Class B",
                                # "Doc class A",
                            ],
                        ],
                    ),
                    "ft.onto.base_ontology.Sentence": SortedList(
                        [
                            [
                                55,
                                70,
                                1234567,
                                1,
                                "Negative",
                                None,
                                None,
                                "Class C",
                                "Class D",
                                "abc",
                            ],
                            [
                                6,
                                9,
                                9999,
                                1,
                                "Postive",
                                "teacher",
                                1,
                                None,
                                None,
                                "cba",
                            ],
                        ],
                    ),
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "Saved objects have unidentified fields, which raise an error.",
            ):
                DataStore.deserialize(tmpfilepath, accept_none=False)


if __name__ == "__main__":
    unittest.main()
