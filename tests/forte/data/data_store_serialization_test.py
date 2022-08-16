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
from typing import Union
import unittest
import tempfile
import os
from sortedcontainers import SortedList
from forte.data.data_store import DataStore
from forte.data.ontology.core import FDict
from ft.onto.base_ontology import Classification

logging.basicConfig(level=logging.DEBUG)


class DataStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_store = DataStore()
        # This test setup changes the order of fields and delete one field of
        # Document entries. It also changes the order of fields and delete one
        # field and add two more fields of Sentence entries.

        # attribute fields and parent entry for Document and Sentence entries
        DataStore._type_attributes = {
            "ft.onto.base_ontology.Document": {
                "attributes": {
                    "sentiment": {"index": 4, "type": (dict, (str, float))},
                    "classifications": {
                        "index": 5,
                        "type": (FDict, (str, Classification)),
                    },
                },
                "parent_entry": "forte.data.ontology.top.Annotation",
            },
            "ft.onto.base_ontology.Sentence": {
                "attributes": {
                    "sentiment": {"index": 4, "type": (dict, (str, float))},
                    "speaker": {"index": 5, "type": (Union, (str, type(None)))},
                    "part_id": {"index": 6, "type": (Union, (int, type(None)))},
                    "classification_test": {
                        "index": 7,
                        "type": (dict, (str, float)),
                    },
                    "classifications": {
                        "index": 8,
                        "type": (FDict, (str, Classification)),
                    },
                    "temp": {"index": 9, "type": (Union, (str, type(None)))},
                },
                "parent_entry": "forte.data.ontology.top.Annotation",
            },
            "forte.data.ontology.top.Group": {
                "attributes": {},
                "parent_entry": "forte.data.ontology.core.BaseGroup",
            },
            "forte.data.ontology.top.Link": {
                "attributes": {},
                "parent_entry": "forte.data.ontology.core.BaseLink",
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
            "forte.data.ontology.top.Group": [
                [
                    "ft.onto.base_ontology.Sentence",
                    [9999, 1234567],
                    10123,
                    "forte.data.ontology.top.Group",
                ],
                [
                    "ft.onto.base_ontology.Document",
                    [1234, 3456],
                    23456,
                    "forte.data.ontology.top.Group",
                ],
            ],
            "forte.data.ontology.top.Link": [
                [
                    9999,
                    1234,
                    88888,
                    "forte.data.ontology.top.Link",
                ],
            ],
        }

        self.data_store._DataStore__tid_ref_dict = {
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

        self.data_store._DataStore__tid_idx_dict = {
            10123: ["forte.data.ontology.top.Group", 0],
            23456: ["forte.data.ontology.top.Group", 1],
            88888: ["forte.data.ontology.top.Link", 0],
        }

    def test_save_attribute_pickle(self):
        # test serialize with save_attribute and deserialize with/without
        # check_attribute and accept_unknown_attribute
        with tempfile.TemporaryDirectory() as tempdir:
            tmpfilepath = os.path.join(tempdir, "temp.txt")
            self.data_store.serialize(
                tmpfilepath,
                serialize_method="json",
                save_attribute=True,
                indent=2,
            )

            DataStore._type_attributes = {
                "ft.onto.base_ontology.Document": {
                    "attributes": {
                        "document_class": {"index": 4, "type": (list, (str,))},
                        "sentiment": {"index": 5, "type": (dict, (str, float))},
                        "classifications": {
                            "index": 6,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "ft.onto.base_ontology.Sentence": {
                    "attributes": {
                        "speaker": {
                            "index": 4,
                            "type": (Union, (str, type(None))),
                        },
                        "part_id": {
                            "index": 5,
                            "type": (Union, (int, type(None))),
                        },
                        "sentiment": {"index": 6, "type": (dict, (str, float))},
                        "classification": {
                            "index": 7,
                            "type": (dict, (str, float)),
                        },
                        "classifications": {
                            "index": 8,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "forte.data.ontology.top.Group": {
                    "attributes": {},
                    "parent_entry": "forte.data.ontology.core.BaseGroup",
                },
                "forte.data.ontology.top.Link": {
                    "attributes": {},
                    "parent_entry": "forte.data.ontology.core.BaseLink",
                },
            }

            # test check_attribute
            temp = DataStore.deserialize(
                tmpfilepath, serialize_method="json", check_attribute=True
            )
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
                    "forte.data.ontology.top.Group": [
                        [
                            "ft.onto.base_ontology.Sentence",
                            [9999, 1234567],
                            10123,
                            "forte.data.ontology.top.Group",
                        ],
                        [
                            "ft.onto.base_ontology.Document",
                            [1234, 3456],
                            23456,
                            "forte.data.ontology.top.Group",
                        ],
                    ],
                    "forte.data.ontology.top.Link": [
                        [
                            9999,
                            1234,
                            88888,
                            "forte.data.ontology.top.Link",
                        ],
                    ],
                },
            )
            self.assertEqual(
                temp._DataStore__tid_ref_dict,
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

            self.assertEqual(
                temp._DataStore__tid_idx_dict,
                {
                    10123: ["forte.data.ontology.top.Group", 0],
                    23456: ["forte.data.ontology.top.Group", 1],
                    88888: ["forte.data.ontology.top.Link", 0],
                },
            )

            temp = DataStore.deserialize(
                tmpfilepath, serialize_method="json", check_attribute=False
            )
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
                        ],
                    ),
                    "forte.data.ontology.top.Group": [
                        [
                            "ft.onto.base_ontology.Sentence",
                            [9999, 1234567],
                            10123,
                            "forte.data.ontology.top.Group",
                        ],
                        [
                            "ft.onto.base_ontology.Document",
                            [1234, 3456],
                            23456,
                            "forte.data.ontology.top.Group",
                        ],
                    ],
                    "forte.data.ontology.top.Link": [
                        [
                            9999,
                            1234,
                            88888,
                            "forte.data.ontology.top.Link",
                        ],
                    ],
                },
            )
            self.assertEqual(
                temp._DataStore__tid_ref_dict,
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

            self.assertEqual(
                temp._DataStore__tid_idx_dict,
                {
                    10123: ["forte.data.ontology.top.Group", 0],
                    23456: ["forte.data.ontology.top.Group", 1],
                    88888: ["forte.data.ontology.top.Link", 0],
                },
            )

            # test check_attribute with accept_unknown_attribute = False
            with self.assertRaisesRegex(
                ValueError,
                "Saved ft.onto.base_ontology.Document objects have unidentified"
                " fields at indices 4, which raise an error.",
            ):
                DataStore.deserialize(
                    tmpfilepath,
                    serialize_method="json",
                    check_attribute=True,
                    accept_unknown_attribute=False,
                )

    def test_fast_pickle(self):
        # test serialize without save_attribute and deserialize with/without
        # check_attribute and accept_unknown_attribute
        with tempfile.TemporaryDirectory() as tempdir:
            tmpfilepath = os.path.join(tempdir, "temp.txt")
            self.data_store.serialize(
                tmpfilepath, serialize_method="json", save_attribute=False
            )
            DataStore._type_attributes = {
                "ft.onto.base_ontology.Document": {
                    "attributes": {
                        "document_class": {"index": 4, "type": (list, (str,))},
                        "sentiment": {"index": 5, "type": (dict, (str, float))},
                        "classifications": {
                            "index": 6,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "ft.onto.base_ontology.Sentence": {
                    "attributes": {
                        "speaker": {
                            "index": 4,
                            "type": (Union, (str, type(None))),
                        },
                        "part_id": {
                            "index": 5,
                            "type": (Union, (int, type(None))),
                        },
                        "sentiment": {"index": 6, "type": (dict, (str, float))},
                        "classification": {
                            "index": 7,
                            "type": (dict, (str, float)),
                        },
                        "classifications": {
                            "index": 8,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "forte.data.ontology.top.Group": {
                    "attributes": {},
                    "parent_entry": "forte.data.ontology.core.BaseGroup",
                },
                "forte.data.ontology.top.Link": {
                    "attributes": {},
                    "parent_entry": "forte.data.ontology.core.BaseLink",
                },
            }
            temp = DataStore.deserialize(
                tmpfilepath, serialize_method="json", check_attribute=False
            )
            self.assertEqual(temp._type_attributes, DataStore._type_attributes)
            self.assertEqual(
                temp._DataStore__tid_ref_dict,
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

            # test check_attribute without save_attribute
            with self.assertRaisesRegex(
                ValueError,
                "The serialized object that you want to deserialize does not support check_attribute.",
            ):
                DataStore.deserialize(
                    tmpfilepath, serialize_method="json", check_attribute=True
                )

    def test_delete_serialize(self):
        self.data_store.delete_entry(4567)
        self.data_store.delete_entry(10123)
        with tempfile.TemporaryDirectory() as tempdir:
            tmpfilepath = os.path.join(tempdir, "temp.txt")
            self.data_store.serialize(
                tmpfilepath,
                serialize_method="json",
                save_attribute=True,
                indent=2,
            )
            DataStore._type_attributes = {
                "ft.onto.base_ontology.Document": {
                    "attributes": {
                        "document_class": {"index": 4, "type": (list, (str,))},
                        "sentiment": {"index": 5, "type": (dict, (str, float))},
                        "classifications": {
                            "index": 6,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "ft.onto.base_ontology.Sentence": {
                    "attributes": {
                        "speaker": {
                            "index": 4,
                            "type": (Union, (str, type(None))),
                        },
                        "part_id": {
                            "index": 5,
                            "type": (Union, (int, type(None))),
                        },
                        "sentiment": {"index": 6, "type": (dict, (str, float))},
                        "classification": {
                            "index": 7,
                            "type": (dict, (str, float)),
                        },
                        "classifications": {
                            "index": 8,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "forte.data.ontology.top.Group": {
                    "attributes": {},
                    "parent_entry": "forte.data.ontology.core.BaseGroup",
                },
                "forte.data.ontology.top.Link": {
                    "attributes": {},
                    "parent_entry": "forte.data.ontology.core.BaseLink",
                },
            }
            temp = DataStore.deserialize(
                tmpfilepath, serialize_method="json", check_attribute=True
            )
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
                    "forte.data.ontology.top.Group": [
                        [
                            "ft.onto.base_ontology.Document",
                            [1234, 3456],
                            23456,
                            "forte.data.ontology.top.Group",
                        ],
                    ],
                    "forte.data.ontology.top.Link": [
                        [
                            9999,
                            1234,
                            88888,
                            "forte.data.ontology.top.Link",
                        ],
                    ],
                },
            )
            self.assertEqual(
                temp._DataStore__tid_ref_dict,
                {
                    1234: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][0],
                    3456: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][1],
                    5678: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][2],
                    7890: temp._DataStore__elements[
                        "ft.onto.base_ontology.Document"
                    ][3],
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

            self.assertEqual(
                temp._DataStore__tid_idx_dict,
                {
                    23456: ["forte.data.ontology.top.Group", 0],
                    88888: ["forte.data.ontology.top.Link", 0],
                },
            )


if __name__ == "__main__":
    unittest.main()
