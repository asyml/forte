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
import timeit
import tempfile
import os
from sortedcontainers import SortedList
from forte.data.data_store import DataStore

logging.basicConfig(level=logging.DEBUG)


class DataStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_store = DataStore()
        # attribute fields and parent entry for Document and Sentence entries
        DataStore._type_attributes = {
            "ft.onto.base_ontology.Document": {
                "attributes": {
                    "sentiment": 4,
                    "classifications": 5,
                },
                "parent_entry": "forte.data.ontology.top.Annotation",
            },
            "ft.onto.base_ontology.Sentence": {
                "attributes": {
                    "sentiment": 4,
                    "speaker": 5,
                    "part_id": 6,
                    "classification_test": 7,
                    "classifications": 8,
                    "temp": 9,
                },
                "parent_entry": "forte.data.ontology.top.Annotation",
            },
        }

        # Document entries have tid 1234, 3456, 4567, 5678, 7890.
        # Sentence entries have tid 9999, 1234567, 100, 5000.
        self.data_store._DataStore__elements = {
            "ft.onto.base_ontology.Document": SortedList(
                [
                    [
                        0,
                        5,
                        1234,
                        "ft.onto.base_ontology.Document",
                        "Positive",
                        None,
                    ],
                    [
                        10,
                        25,
                        3456,
                        "ft.onto.base_ontology.Document",
                        "Negative",
                        "Class B",
                    ],
                    [
                        15,
                        20,
                        4567,
                        "ft.onto.base_ontology.Document",
                        "Positive",
                        "Class C",
                    ],
                    [
                        20,
                        25,
                        5678,
                        "ft.onto.base_ontology.Document",
                        "Neutral",
                        "Class D",
                    ],
                    [
                        40,
                        55,
                        7890,
                        "ft.onto.base_ontology.Document",
                        "Very Positive",
                        "Class E",
                    ],
                ],
            ),
            "ft.onto.base_ontology.Sentence": SortedList(
                [
                    [
                        6,
                        9,
                        9999,
                        "ft.onto.base_ontology.Sentence",
                        "Positive",
                        "teacher",
                        1,
                        None,
                        None,
                        "cba",
                    ],
                    [
                        55,
                        70,
                        1234567,
                        "ft.onto.base_ontology.Sentence",
                        "Negative",
                        None,
                        None,
                        "Class C",
                        "Class D",
                        "abc",
                    ],
                    [
                        60,
                        90,
                        100,
                        "ft.onto.base_ontology.Sentence",
                        "Positive",
                        "student",
                        2,
                        "testA",
                        "class1",
                        "bad",
                    ],
                    [
                        65,
                        90,
                        5000,
                        "ft.onto.base_ontology.Sentence",
                        "Positive",
                        "TA",
                        3,
                        "testB",
                        "class2",
                        "good",
                    ],
                ],
            ),
        }

        self.data_store._DataStore__entry_dict = {
            1234: self.data_store._DataStore__elements[
                "ft.onto.base_ontology.Document"
            ][0],
            3456: self.data_store._DataStore__elements[
                "ft.onto.base_ontology.Document"
            ][1],
            4567: self.data_store._DataStore__elements[
                "ft.onto.base_ontology.Document"
            ][2],
            5678: self.data_store._DataStore__elements[
                "ft.onto.base_ontology.Document"
            ][3],
            7890: self.data_store._DataStore__elements[
                "ft.onto.base_ontology.Document"
            ][4],
            9999: self.data_store._DataStore__elements[
                "ft.onto.base_ontology.Sentence"
            ][0],
            1234567: self.data_store._DataStore__elements[
                "ft.onto.base_ontology.Sentence"
            ][1],
            100: self.data_store._DataStore__elements[
                "ft.onto.base_ontology.Sentence"
            ][2],
            5000: self.data_store._DataStore__elements[
                "ft.onto.base_ontology.Sentence"
            ][3],
        }

    def test_save_attribute_pickle(self):
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = "temp/"
            tmpfilepath = os.path.join(tempdir, "temp.txt")
            a = timeit.timeit()
            self.data_store.serialize(
                tmpfilepath, serialize_method="json", save_attribute=True
            )
            b = timeit.timeit()

            DataStore._type_attributes = {
                "ft.onto.base_ontology.Document": {
                    "attributes": {
                        "document_class": 4,
                        "sentiment": 5,
                        "classifications": 6,
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "ft.onto.base_ontology.Sentence": {
                    "attributes": {
                        "speaker": 4,
                        "part_id": 5,
                        "sentiment": 6,
                        "classification": 7,
                        "classifications": 8,
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
            }
            c = timeit.timeit()
            temp = DataStore.deserialize(
                tmpfilepath, serialize_method="json", check_attribute=True
            )
            d = timeit.timeit()
            self.assertEqual(temp._type_attributes, DataStore._type_attributes)
            self.assertEqual(
                temp._DataStore__elements,
                {
                    "ft.onto.base_ontology.Document": SortedList(
                        [
                            [
                                0,
                                5,
                                1234,
                                "ft.onto.base_ontology.Document",
                                None,
                                "Positive",
                                None,
                            ],
                            [
                                10,
                                25,
                                3456,
                                "ft.onto.base_ontology.Document",
                                None,
                                "Negative",
                                "Class B",
                            ],
                            [
                                15,
                                20,
                                4567,
                                "ft.onto.base_ontology.Document",
                                None,
                                "Positive",
                                "Class C",
                            ],
                            [
                                20,
                                25,
                                5678,
                                "ft.onto.base_ontology.Document",
                                None,
                                "Neutral",
                                "Class D",
                            ],
                            [
                                40,
                                55,
                                7890,
                                "ft.onto.base_ontology.Document",
                                None,
                                "Very Positive",
                                "Class E",
                            ],
                        ],
                    ),
                    "ft.onto.base_ontology.Sentence": SortedList(
                        [
                            [
                                6,
                                9,
                                9999,
                                "ft.onto.base_ontology.Sentence",
                                "teacher",
                                1,
                                "Positive",
                                None,
                                None,
                            ],
                            [
                                55,
                                70,
                                1234567,
                                "ft.onto.base_ontology.Sentence",
                                None,
                                None,
                                "Negative",
                                None,
                                "Class D",
                            ],
                            [
                                60,
                                90,
                                100,
                                "ft.onto.base_ontology.Sentence",
                                "student",
                                2,
                                "Positive",
                                None,
                                "class1",
                            ],
                            [
                                65,
                                90,
                                5000,
                                "ft.onto.base_ontology.Sentence",
                                "TA",
                                3,
                                "Positive",
                                None,
                                "class2",
                            ],
                        ],
                    ),
                },
            )
            self.assertEqual(
                temp._DataStore__entry_dict,
                {
                    1234: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][0],
                    3456: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][1],
                    4567: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][2],
                    5678: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][3],
                    7890: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][4],
                    9999: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][0],
                    1234567: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][1],
                    100: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][2],
                    5000: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][3],
                },
            )
            print(f"serialization with save attribute time cost {b-a}")
            print(f"deserialization with check attribute time cost {d-c}")

            e = timeit.timeit()
            temp = DataStore.deserialize(
                tmpfilepath, serialize_method="json", check_attribute=False
            )
            f = timeit.timeit()
            self.assertEqual(temp._type_attributes, DataStore._type_attributes)
            self.assertEqual(
                temp._DataStore__elements,
                {
                    "ft.onto.base_ontology.Document": SortedList(
                        [
                            [
                                0,
                                5,
                                1234,
                                "ft.onto.base_ontology.Document",
                                "Positive",
                                None,
                            ],
                            [
                                10,
                                25,
                                3456,
                                "ft.onto.base_ontology.Document",
                                "Negative",
                                "Class B",
                            ],
                            [
                                15,
                                20,
                                4567,
                                "ft.onto.base_ontology.Document",
                                "Positive",
                                "Class C",
                            ],
                            [
                                20,
                                25,
                                5678,
                                "ft.onto.base_ontology.Document",
                                "Neutral",
                                "Class D",
                            ],
                            [
                                40,
                                55,
                                7890,
                                "ft.onto.base_ontology.Document",
                                "Very Positive",
                                "Class E",
                            ],
                        ],
                    ),
                    "ft.onto.base_ontology.Sentence": SortedList(
                        [
                            [
                                6,
                                9,
                                9999,
                                "ft.onto.base_ontology.Sentence",
                                "Positive",
                                "teacher",
                                1,
                                None,
                                None,
                                "cba",
                            ],
                            [
                                55,
                                70,
                                1234567,
                                "ft.onto.base_ontology.Sentence",
                                "Negative",
                                None,
                                None,
                                "Class C",
                                "Class D",
                                "abc",
                            ],
                            [
                                60,
                                90,
                                100,
                                "ft.onto.base_ontology.Sentence",
                                "Positive",
                                "student",
                                2,
                                "testA",
                                "class1",
                                "bad",
                            ],
                            [
                                65,
                                90,
                                5000,
                                "ft.onto.base_ontology.Sentence",
                                "Positive",
                                "TA",
                                3,
                                "testB",
                                "class2",
                                "good",
                            ],
                        ]
                    ),
                },
            )
            self.assertEqual(
                temp._DataStore__entry_dict,
                {
                    1234: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][0],
                    3456: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][1],
                    4567: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][2],
                    5678: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][3],
                    7890: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][4],
                    9999: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][0],
                    1234567: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][1],
                    100: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][2],
                    5000: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][3],
                },
            )
            print(f"deserialization without check attribute time cost {f-e}")

            with self.assertRaisesRegex(
                ValueError,
                "Saved objects have unidentified fields, which raise an error.",
            ):
                DataStore.deserialize(
                    tmpfilepath,
                    serialize_method="json",
                    check_attribute=True,
                    accept_none=False,
                )

    def test_fast_pickle(self):
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = "temp/"
            tmpfilepath = os.path.join(tempdir, "temp2.txt")
            a = timeit.timeit()
            self.data_store.serialize(
                tmpfilepath, serialize_method="json", save_attribute=False
            )
            b = timeit.timeit()
            DataStore._type_attributes = {
                "ft.onto.base_ontology.Document": {
                    "attributes": {
                        "document_class": 4,
                        "sentiment": 5,
                        "classifications": 6,
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "ft.onto.base_ontology.Sentence": {
                    "attributes": {
                        "speaker": 4,
                        "part_id": 5,
                        "sentiment": 6,
                        "classification": 7,
                        "classifications": 8,
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
            }
            c = timeit.timeit()
            temp = DataStore.deserialize(
                tmpfilepath, serialize_method="json", check_attribute=False
            )
            d = timeit.timeit()
            print(f"serialization without save attribute time cost {b-a}")
            self.assertEqual(temp._type_attributes, DataStore._type_attributes)
            self.assertEqual(
                temp._DataStore__entry_dict,
                {
                    1234: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][0],
                    3456: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][1],
                    4567: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][2],
                    5678: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][3],
                    7890: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][4],
                    9999: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][0],
                    1234567: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][1],
                    100: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][2],
                    5000: temp._DataStore__elements[
                        "ft.onto.base_ontology.Sentence"
                    ][3],
                },
            )
            print(f"deserialization without check attribute time cost {d-c}")

            with self.assertRaisesRegex(
                ValueError, "Saved object does not support check_attribute."
            ):
                DataStore.deserialize(tmpfilepath, serialize_method="json")


if __name__ == "__main__":
    unittest.main()
