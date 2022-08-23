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
from typing import Union, Any
import unittest
import tempfile
import os
from sortedcontainers import SortedList
from forte.data.data_store import DataStore
from forte.common import constants
from forte.data.ontology.core import FDict, FList, Entry
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
                    "begin": {"index": 2, "type": (type(None), (int,))},
                    "end": {"index": 3, "type": (type(None), (int,))},
                    "payload_idx": {"index": 4, "type": (type(None), (int,))},
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
                    "begin": {"index": 2, "type": (type(None), (int,))},
                    "end": {"index": 3, "type": (type(None), (int,))},
                    "payload_idx": {"index": 4, "type": (type(None), (int,))},
                    "sentiment": {"index": 5, "type": (dict, (str, float))},
                    "speaker": {"index": 6, "type": (Union, (str, type(None)))},
                    "part_id": {"index": 7, "type": (Union, (int, type(None)))},
                    "classification_test": {
                        "index": 8,
                        "type": (dict, (str, float)),
                    },
                    "classifications": {
                        "index": 9,
                        "type": (FDict, (str, Classification)),
                    },
                    "temp": {"index": 10, "type": (Union, (str, type(None)))},
                },
                "parent_entry": "forte.data.ontology.top.Annotation",
            },
            "forte.data.ontology.top.Group": {
                "attributes": {
                    'members': {'type': (FList, (Entry,)), 'index': 2},
                    'member_type': {'type': (type(None), (str,)), 'index': 3}
                },
                "parent_entry": "forte.data.ontology.core.BaseGroup",
            },
            "forte.data.ontology.top.Link": {
                "attributes": {
                    'parent_type': {'type': (type(None), (str,)), 'index': 2},
                    'child_type': {'type': (type(None), (str,)), 'index': 3},
                    'parent': {'type': (Union, (int, type(None))), 'index': 4},
                    'child': {'type': (Union, (int, type(None))), 'index': 5}
                },
                "parent_entry": "forte.data.ontology.core.BaseLink",
            },
        }

        self.sorting_fn = lambda s: (
            s[2],
            s[3],
        )

        # Document entries have tid 1234, 3456, 4567, 5678, 7890.
        # Sentence entries have tid 9999, 1234567, 100, 5000.
        self.data_store._DataStore__elements = {
            "ft.onto.base_ontology.Document": SortedList(
                [
                    [
                        1234,
                        "ft.onto.base_ontology.Document",
                        0,
                        5,
                        0,
                        "Positive",
                        None,
                    ],
                    [
                        3456,
                        "ft.onto.base_ontology.Document",
                        10,
                        25,
                        0,
                        "Negative",
                        "Class B",
                    ],
                    [
                        4567,
                        "ft.onto.base_ontology.Document",
                        15,
                        20,
                        0,
                        "Positive",
                        "Class C",
                    ],
                    [
                        5678,
                        "ft.onto.base_ontology.Document",
                        20,
                        25,
                        0,
                        "Neutral",
                        "Class D",
                    ],
                    [
                        7890,
                        "ft.onto.base_ontology.Document",
                        40,
                        55,
                        0,
                        "Very Positive",
                        "Class E",
                    ],
                ],
            key=self.sorting_fn),
            "ft.onto.base_ontology.Sentence": SortedList(
                [
                    [
                        9999,
                        "ft.onto.base_ontology.Sentence",
                        6,
                        9,
                        0,
                        "Positive",
                        "teacher",
                        1,
                        None,
                        None,
                        "cba",
                    ],
                    [
                        1234567,
                        "ft.onto.base_ontology.Sentence",
                        55,
                        70,
                        0,
                        "Negative",
                        None,
                        None,
                        "Class C",
                        "Class D",
                        "abc",
                    ],
                    [
                        100,
                        "ft.onto.base_ontology.Sentence",
                        60,
                        90,
                        0,
                        "Positive",
                        "student",
                        2,
                        "testA",
                        "class1",
                        "bad",
                    ],
                    [
                        5000,
                        "ft.onto.base_ontology.Sentence",
                        65,
                        90,
                        0,
                        "Positive",
                        "TA",
                        3,
                        "testB",
                        "class2",
                        "good",
                    ],
                ],
            key=self.sorting_fn),
            "forte.data.ontology.top.Group": [
                [
                    10123,
                    "forte.data.ontology.top.Group",
                    [9999, 1234567],
                    "ft.onto.base_ontology.Sentence"
                ],
                [
                    23456,
                    "forte.data.ontology.top.Group",
                    [1234, 3456],
                    "ft.onto.base_ontology.Document"
                ],
            ],
            "forte.data.ontology.top.Link": [
                [
                    88888,
                    "forte.data.ontology.top.Link",
                    "ft.onto.base_ontology.Sentence",
                    "ft.onto.base_ontology.Document",
                    9999,
                    1234,
                    
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
                        "begin": {"index": 2, "type": (type(None), (int,))},
                        "end": {"index": 3, "type": (type(None), (int,))},
                        "payload_idx": {"index": 4, "type": (type(None), (int,))},
                        "document_class": {"index": 5, "type": (list, (str,))},
                        "sentiment": {"index": 6, "type": (dict, (str, float))},
                        "classifications": {
                            "index": 7,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "ft.onto.base_ontology.Sentence": {
                    "attributes": {
                        "begin": {
                            "index": 2,
                            "type": (type(None), (int,))
                        },
                        "end": {
                            "index": 3,
                            "type": (type(None), (int,))
                        },
                        "payload_idx": {
                            "index": 4,
                            "type": (type(None), (int,))
                        },
                        "speaker": {
                            "index": 5,
                            "type": (Union, (str, type(None))),
                        },
                        "part_id": {
                            "index": 6,
                            "type": (Union, (int, type(None))),
                        },
                        "sentiment": {
                            "index": 7, 
                            "type": (dict, (str, float))
                        },
                        "classification": {
                            "index": 8,
                            "type": (dict, (str, float)),
                        },
                        "classifications": {
                            "index": 9,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "forte.data.ontology.top.Group": {
                    "attributes": {
                        'members': {'type': (FList, (Entry,)), 'index': 2},
                        'member_type': {'type': (type(None), (str,)), 'index': 3}
                    },
                    "parent_entry": "forte.data.ontology.core.BaseGroup",
                },
                "forte.data.ontology.top.Link": {
                    "attributes": {
                        'parent_type': {'type': (type(None), (str,)), 'index': 2},
                        'child_type': {'type': (type(None), (str,)), 'index': 3},
                        'parent': {'type': (Union, (int, type(None))), 'index': 4},
                        'child': {'type': (Union, (int, type(None))), 'index': 5}
                    },
                    "parent_entry": "forte.data.ontology.core.BaseLink",
                },
            }

            # test check_attribute
            temp = DataStore.deserialize(
                tmpfilepath, serialize_method="json", check_attribute=True
            )
            self.assertEqual(temp._type_attributes, DataStore._type_attributes)

            self.sorting_fn = lambda s: (
                s[2],
                s[3],
            )

            self.assertEqual(
                temp._DataStore__elements,
                {
                    "ft.onto.base_ontology.Document": SortedList(
                        [
                            [
                                1234,
                                "ft.onto.base_ontology.Document",
                                0,
                                5,
                                0,
                                None,
                                "Positive",
                                None,
                            ],
                            [
                                3456,
                                "ft.onto.base_ontology.Document",
                                10,
                                25,
                                0,
                                None,
                                "Negative",
                                "Class B",
                            ],
                            [
                                4567,
                                "ft.onto.base_ontology.Document",
                                15,
                                20,
                                0,
                                None,
                                "Positive",
                                "Class C",
                            ],
                            [
                                5678,
                                "ft.onto.base_ontology.Document",
                                20,
                                25,
                                0,
                                None,
                                "Neutral",
                                "Class D",
                            ],
                            [
                                7890,
                                "ft.onto.base_ontology.Document",
                                40,
                                55,
                                0,
                                None,
                                "Very Positive",
                                "Class E",
                            ],
                        ],
                    key=self.sorting_fn),
                    "ft.onto.base_ontology.Sentence": SortedList(
                        [
                            [
                                9999,
                                "ft.onto.base_ontology.Sentence",
                                6,
                                9,
                                0,
                                "teacher",
                                1,
                                "Positive",
                                None,
                                None,
                            ],
                            [
                                1234567,
                                "ft.onto.base_ontology.Sentence",
                                55,
                                70,
                                0,
                                None,
                                None,
                                "Negative",
                                None,
                                "Class D",
                            ],
                            [
                                100,
                                "ft.onto.base_ontology.Sentence",
                                60,
                                90,
                                0,
                                "student",
                                2,
                                "Positive",
                                None,
                                "class1",
                            ],
                            [
                                5000,
                                "ft.onto.base_ontology.Sentence",
                                65,
                                90,
                                0,
                                "TA",
                                3,
                                "Positive",
                                None,
                                "class2",
                            ],
                        ],
                    key=self.sorting_fn),
                    "forte.data.ontology.top.Group": [
                        [
                            10123,
                            "forte.data.ontology.top.Group",
                            [9999, 1234567],
                            "ft.onto.base_ontology.Sentence"
                        ],
                        [
                            23456,
                            "forte.data.ontology.top.Group",
                            [1234, 3456],
                            "ft.onto.base_ontology.Document"
                        ],
                    ],
                    "forte.data.ontology.top.Link": [
                        [
                            88888,
                            "forte.data.ontology.top.Link",
                            "ft.onto.base_ontology.Sentence",
                            "ft.onto.base_ontology.Document",
                            9999,
                            1234,
                            
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

            self.sorting_fn = lambda s: (
                s[2],
                s[3],
            )

            self.assertEqual(
                temp._DataStore__elements,
                {
                    "ft.onto.base_ontology.Document": SortedList(
                        [
                            [
                                1234,
                                "ft.onto.base_ontology.Document",
                                0,
                                5,
                                0,
                                "Positive",
                                None,
                            ],
                            [
                                3456,
                                "ft.onto.base_ontology.Document",
                                10,
                                25,
                                0,
                                "Negative",
                                "Class B",
                            ],
                            [
                                4567,
                                "ft.onto.base_ontology.Document",
                                15,
                                20,
                                0,
                                "Positive",
                                "Class C",
                            ],
                            [
                                5678,
                                "ft.onto.base_ontology.Document",
                                20,
                                25,
                                0,
                                "Neutral",
                                "Class D",
                            ],
                            [
                                7890,
                                "ft.onto.base_ontology.Document",
                                40,
                                55,
                                0,
                                "Very Positive",
                                "Class E",
                            ]
                        ],
                        key=self.sorting_fn),
                    "ft.onto.base_ontology.Sentence": SortedList(
                        [
                            [
                                9999,
                                "ft.onto.base_ontology.Sentence",
                                6,
                                9,
                                0,
                                "Positive",
                                "teacher",
                                1,
                                None,
                                None,
                                "cba",
                            ],
                            [
                                1234567,
                                "ft.onto.base_ontology.Sentence",
                                55,
                                70,
                                0,
                                "Negative",
                                None,
                                None,
                                "Class C",
                                "Class D",
                                "abc",

                            ],
                            [
                                100,
                                "ft.onto.base_ontology.Sentence",
                                60,
                                90,
                                0,
                                "Positive",
                                "student",
                                2,
                                "testA",
                                "class1",
                                "bad",
                            ],
                            [
                                5000,
                                "ft.onto.base_ontology.Sentence",
                                65,
                                90,
                                0,
                                "Positive",
                                "TA",
                                3,
                                "testB",
                                "class2",
                                "good",
                            ],
                        ],
                        key=self.sorting_fn
                    ),
                    "forte.data.ontology.top.Group": [
                        [
                            10123,
                            "forte.data.ontology.top.Group",
                            [9999, 1234567],
                            "ft.onto.base_ontology.Sentence"
                        ],
                        [
                            23456,
                            "forte.data.ontology.top.Group",
                            [1234, 3456],
                            "ft.onto.base_ontology.Document"
                        ],
                    ],
                    "forte.data.ontology.top.Link": [
                        [
                            88888,
                            "forte.data.ontology.top.Link",
                            "ft.onto.base_ontology.Sentence",
                            "ft.onto.base_ontology.Document",
                            9999,
                            1234,
                            
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
                " fields at indices 5, which raise an error.",
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
                        "begin": {"index": 2, "type": (type(None), (int,))},
                        "end": {"index": 3, "type": (type(None), (int,))},
                        "payload_idx": {"index": 4, "type": (type(None), (int,))},
                        "document_class": {"index": 5, "type": (list, (str,))},
                        "sentiment": {"index": 6, "type": (dict, (str, float))},
                        "classifications": {
                            "index": 7,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "ft.onto.base_ontology.Sentence": {
                    "attributes": {
                        "begin": {
                            "index": 2,
                            "type": (type(None), (int,))
                        },
                        "end": {
                            "index": 3,
                            "type": (type(None), (int,))
                        },
                        "payload_idx": {
                            "index": 4,
                            "type": (type(None), (int,))
                        },
                        "speaker": {
                            "index": 5,
                            "type": (Union, (str, type(None))),
                        },
                        "part_id": {
                            "index": 6,
                            "type": (Union, (int, type(None))),
                        },
                        "sentiment": {"index": 7, "type": (dict, (str, float))},
                        "classification": {
                            "index": 8,
                            "type": (dict, (str, float)),
                        },
                        "classifications": {
                            "index": 9,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "forte.data.ontology.top.Group": {
                    "attributes": {
                        'members': {'type': (FList, (Entry,)), 'index': 2},
                        'member_type': {'type': (type(None), (str,)), 'index': 3}
                    },
                    "parent_entry": "forte.data.ontology.core.BaseGroup",
                },
                "forte.data.ontology.top.Link": {
                    "attributes": {
                        'parent_type': {'type': (type(None), (str,)), 'index': 2},
                        'child_type': {'type': (type(None), (str,)), 'index': 3},
                        'parent': {'type': (Union, (int, type(None))), 'index': 4},
                        'child': {'type': (Union, (int, type(None))), 'index': 5}
                    },
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
                        "begin": {"index": 2, "type": (type(None), (int,))},
                        "end": {"index": 3, "type": (type(None), (int,))},
                        "payload_idx": {"index": 4, "type": (type(None), (int,))},
                        "document_class": {"index": 5, "type": (list, (str,))},
                        "sentiment": {"index": 6, "type": (dict, (str, float))},
                        "classifications": {
                            "index": 7,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "ft.onto.base_ontology.Sentence": {
                    "attributes": {
                        "begin": {
                            "index": 2,
                            "type": (type(None), (int,))
                        },
                        "end": {
                            "index": 3,
                            "type": (type(None), (int,))
                        },
                        "payload_idx": {
                            "index": 4,
                            "type": (type(None), (int,))
                        },
                        "speaker": {
                            "index": 5,
                            "type": (Union, (str, type(None))),
                        },
                        "part_id": {
                            "index": 6,
                            "type": (Union, (int, type(None))),
                        },
                        "sentiment": {"index": 7, "type": (dict, (str, float))},
                        "classification": {
                            "index": 8,
                            "type": (dict, (str, float)),
                        },
                        "classifications": {
                            "index": 9,
                            "type": (FDict, (str, Classification)),
                        },
                    },
                    "parent_entry": "forte.data.ontology.top.Annotation",
                },
                "forte.data.ontology.top.Group": {
                    "attributes": {
                        'members': {'type': (FList, (Entry,)), 'index': 2},
                        'member_type': {'type': (type(None), (str,)), 'index': 3}
                    },
                    "parent_entry": "forte.data.ontology.core.BaseGroup",
                },
                "forte.data.ontology.top.Link": {
                    "attributes": {
                        'parent_type': {'type': (type(None), (str,)), 'index': 2},
                        'child_type': {'type': (type(None), (str,)), 'index': 3},
                        'parent': {'type': (Union, (int, type(None))), 'index': 4},
                        'child': {'type': (Union, (int, type(None))), 'index': 5}
                    },
                    "parent_entry": "forte.data.ontology.core.BaseLink",
                },
            }
            temp = DataStore.deserialize(
                tmpfilepath, serialize_method="json", check_attribute=True
            )
            self.assertEqual(temp._type_attributes, DataStore._type_attributes)

            self.sorting_fn = lambda s: (
                s[2],
                s[3],
            )

            self.assertEqual(
                temp._DataStore__elements,
                {
                    "ft.onto.base_ontology.Document": SortedList(
                        [
                            [
                                1234,
                                "ft.onto.base_ontology.Document",
                                0,
                                5,
                                0,
                                None,
                                "Positive",
                                None,
                            ],
                            [
                                3456,
                                "ft.onto.base_ontology.Document",
                                10,
                                25,
                                0,
                                None,
                                "Negative",
                                "Class B",
                            ],
                            [
                                5678,
                                "ft.onto.base_ontology.Document",
                                20,
                                25,
                                0,
                                None,
                                "Neutral",
                                "Class D",
                            ],
                            [
                                7890,
                                "ft.onto.base_ontology.Document",
                                40,
                                55,
                                0,
                                None,
                                "Very Positive",
                                "Class E",
                            ],
                        ],
                    key=self.sorting_fn),
                    "ft.onto.base_ontology.Sentence": SortedList(
                        [
                            [
                                9999,
                                "ft.onto.base_ontology.Sentence",
                                6,
                                9,
                                0,
                                "teacher",
                                1,
                                "Positive",
                                None,
                                None,
                            ],
                            [
                                1234567,
                                "ft.onto.base_ontology.Sentence",
                                55,
                                70,
                                0,
                                None,
                                None,
                                "Negative",
                                None,
                                "Class D",
                            ],
                            [
                                100,
                                "ft.onto.base_ontology.Sentence",
                                60,
                                90,
                                0,
                                "student",
                                2,
                                "Positive",
                                None,
                                "class1",
                            ],
                            [
                                5000,
                                "ft.onto.base_ontology.Sentence",
                                65,
                                90,
                                0,
                                "TA",
                                3,
                                "Positive",
                                None,
                                "class2",

                            ],
                        ],
                    key=self.sorting_fn),
                    "forte.data.ontology.top.Group": [
                        [
                            23456,
                            "forte.data.ontology.top.Group",
                            [1234, 3456],
                            "ft.onto.base_ontology.Document"
                        ],
                    ],
                    "forte.data.ontology.top.Link": [
                        [
                            88888,
                            "forte.data.ontology.top.Link",
                            "ft.onto.base_ontology.Sentence",
                            "ft.onto.base_ontology.Document",
                            9999,
                            1234,
                            
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
