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
from typing import Optional, Dict
from dataclasses import dataclass
from forte.data.data_store import DataStore
from forte.data.ontology.top import Annotation, Generics
from forte.data.data_pack import DataPack


logging.basicConfig(level=logging.DEBUG)


@dataclass
class TokenTest(Annotation):
    """
    A span based annotation :class:`Tokentest`, used to represent a token or a word.
    Attributes:
        pos (Optional[str]):
        ud_xpos (Optional[str]):
        lemma (Optional[str]):
        chunk (Optional[str]):
        ner (Optional[str]):
        sense (Optional[str]):
        is_root (Optional[bool]):
        ud_features (Dict[str, str]):
        ud_misc (Dict[str, str]):
    """

    pos: Optional[str]
    ud_xpos: Optional[str]
    lemma: Optional[str]
    chunk: Optional[str]
    ner: Optional[str]
    sense: Optional[str]
    is_root: Optional[bool]
    ud_features: Dict[str, str]
    ud_misc: Dict[str, str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos: Optional[str] = None
        self.ud_xpos: Optional[str] = None
        self.lemma: Optional[str] = None
        self.chunk: Optional[str] = None
        self.ner: Optional[str] = None
        self.sense: Optional[str] = None
        self.is_root: Optional[bool] = None
        self.ud_features: Dict[str, str] = dict()
        self.ud_misc: Dict[str, str] = dict()


@dataclass
class TitleTest(Annotation):
    """
    A span based annotation `Title`, normally used to represent a title.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class MetricTest(Generics):
    """
    A base metric entity, all metric entities should inherit from it.
    Attributes:
        metric_name (Optional[str]):
    """

    metric_name: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.metric_name: Optional[str] = None


@dataclass
class SingleMetricTest(MetricTest):
    """
    A single metric entity, used to present a metric of one float (e.g. accuracy).
    Attributes:
        value (Optional[float]):
    """

    value: Optional[float]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.value: Optional[float] = None


class DataStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_store = DataStore()
        # attribute fields for Document and Sentence entries
        self.data_store._type_attributes = {
            "ft.onto.base_ontology.Document": {
                "attributes": {
                    "document_class": 4,
                    "sentiment": 5,
                    "classifications": 6,
                },
                "parent_entry": None,
            },
            "ft.onto.base_ontology.Sentence": {
                "attributes": {
                    "speaker": 4,
                    "part_id": 5,
                    "sentiment": 6,
                    "classification": 7,
                    "classifications": 8,
                },
                "parent_entry": None,
            },
        }
        # The order is [Document, Sentence]. Initialize 2 entries in each list.
        # Document entries have tid 1234, 3456.
        # Sentence entries have tid 9999, 1234567.
        # The type id for Document is 0, Sentence is 1.

        self.data_store._DataStore__elements = {
            "ft.onto.base_ontology.Document":
                SortedList([
                    [
                        0,
                        5,
                        1234,
                        "ft.onto.base_ontology.Document",
                        None,
                        "Postive",
                        None,
                    ],
                    [
                        10,
                        25,
                        3456,
                        "ft.onto.base_ontology.Document",
                        "Doc class A",
                        "Negative",
                        "Class B",
                    ],
                ]),
            "ft.onto.base_ontology.Sentence":
                SortedList([
                    [
                        6,
                        9,
                        9999,
                        "ft.onto.base_ontology.Sentence",
                        "teacher",
                        1,
                        "Postive",
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
                        "Class C",
                        "Class D",
                    ],
                ]),
            # empty list corresponds to Entry, test only
            "forte.data.ontology.core.Entry": SortedList([]),
            "ft.onto.base_ontology.Phrase": SortedList([
                [
                    1,
                    [9999, 1234567],
                    10123,
                    "ft.onto.base_ontology.Phrase",
                    "Sentence",
                    0,
                ]
            ]),
        }
        self.data_store._DataStore__entry_dict = {
            1234: [
                0,
                5,
                1234,
                "ft.onto.base_ontology.Document",
                None,
                "Postive",
                None,
            ],
            3456: [
                10,
                25,
                3456,
                "ft.onto.base_ontology.Document",
                "Doc class A",
                "Negative",
                "Class B",
            ],
            9999: [
                6,
                9,
                9999,
                "ft.onto.base_ontology.Sentence",
                "teacher",
                1,
                "Postive",
                None,
                None,
            ],
            1234567: [
                55,
                70,
                1234567,
                "ft.onto.base_ontology.Sentence",
                None,
                None,
                "Negative",
                "Class C",
                "Class D",
            ],
            10123: [
                1,
                [9999, 1234567],
                10123,
                "ft.onto.base_ontology.Phrase",
                "Sentence",
                0,
            ]
        }

    def test_get_type_info(self):
        # initialize
        empty_data_store = DataStore()

        # test add new type to _type_attributes
        doc_attr_dict = empty_data_store._get_type_info("ft.onto.base_ontology.Document")
        empty_data_store._get_type_info("ft.onto.base_ontology.Sentence")
        self.assertEqual(len(empty_data_store._DataStore__elements), 0)
        self.assertEqual(empty_data_store._type_attributes["ft.onto.base_ontology.Sentence"],
                        self.data_store._type_attributes["ft.onto.base_ontology.Sentence"])
        self.assertEqual(empty_data_store._type_attributes["ft.onto.base_ontology.Document"],
                        self.data_store._type_attributes["ft.onto.base_ontology.Document"])
        # test the return value
        self.assertEqual(doc_attr_dict, empty_data_store._type_attributes["ft.onto.base_ontology.Document"]["attributes"])
        # test add invalid type
        with self.assertRaisesRegex(
            ValueError, "Class not found in invalid.Type"
        ):
            self.data_store._get_type_info("invalid.Type")
        self.assertTrue("invalid.Type" not in self.data_store._type_attributes)

    def test_add_annotation_raw(self):
        # test add Document entry
        self.data_store.add_annotation_raw("ft.onto.base_ontology.Document", 1, 5)
        # test add Sentence entry
        self.data_store.add_annotation_raw("ft.onto.base_ontology.Sentence", 5, 8)
        num_doc = len(self.data_store._DataStore__elements["ft.onto.base_ontology.Document"])
        num_sent = len(self.data_store._DataStore__elements["ft.onto.base_ontology.Sentence"])

        self.assertEqual(num_doc, 3)
        self.assertEqual(num_sent, 3)
        self.assertEqual(len(self.data_store._DataStore__entry_dict), 7)

        # test add new annotation type
        self.data_store.add_annotation_raw("ft.onto.base_ontology.EntityMention", 10, 12)
        num_phrase = len(self.data_store._DataStore__elements["ft.onto.base_ontology.EntityMention"])
        self.assertEqual(num_phrase, 1)
        self.assertEqual(len(self.data_store._type_attributes), 3)
        self.assertEqual(len(self.data_store._DataStore__entry_dict), 8)

    def test_get_attribute(self):
        speaker = self.data_store.get_attribute(9999, "speaker")
        classifications = self.data_store.get_attribute(3456, "classifications")

        self.assertEqual(speaker, "teacher")
        self.assertEqual(classifications, "Class B")

        # Entry with such tid does not exist
        with self.assertRaisesRegex(KeyError, "Entry with tid 1111 not found."):
            self.data_store.get_attribute(1111, "speaker")

        # Get attribute field that does not exist
        with self.assertRaisesRegex(
            KeyError, "ft.onto.base_ontology.Sentence has no class attribute."
        ):
            self.data_store.get_attribute(9999, "class")

    def test_set_attribute(self):
        # change attribute
        self.data_store.set_attribute(9999, "speaker", "student")
        # set attribute with originally none value
        self.data_store.set_attribute(1234, "document_class", "Class D")
        speaker = self.data_store.get_attribute(9999, "speaker")
        doc_class = self.data_store.get_attribute(1234, "document_class")

        self.assertEqual(speaker, "student")
        self.assertEqual(doc_class, "Class D")

        # Entry with such tid does not exist
        with self.assertRaisesRegex(KeyError, "Entry with tid 1111 not found."):
            self.data_store.set_attribute(1111, "speaker", "human")

        # Set attribute field that does not exist
        with self.assertRaisesRegex(
            KeyError, "ft.onto.base_ontology.Sentence has no speak attribute."
        ):
            self.data_store.set_attribute(9999, "speak", "human")

    def test_get_entry(self):
        # sent = self.data_store.get_entry(1234567)
        # self.assertEqual(
        #     sent[0],
        #     [
        #         55,
        #         70,
        #         1234567,
        #         1,
        #         None,
        #         None,
        #         "Negative",
        #         "Class C",
        #         "Class D",
        #     ],
        # )

        # # Entry with such tid does not exist
        # with self.assertRaises(ValueError):
        #     for doc in self.data_store.get_entry(1111):
        #         print(doc)
        pass

    def test_get(self):
        # get document entries
        instances = list(self.data_store.get("ft.onto.base_ontology.Document"))
        self.assertEqual(len(instances), 2)
        # check tid
        self.assertEqual(instances[0][2], 1234)
        self.assertEqual(instances[1][2], 3456)

        # get all entries
        instances = list(self.data_store.get("forte.data.ontology.core.Entry"))
        self.assertEqual(len(instances), 5)

        # get entries without subclasses
        instances = list(self.data_store.get("forte.data.ontology.core.Entry", include_sub_type=False))
        self.assertEqual(len(instances), 0)

    def test_delete_entry(self):
        # delete annotation
        # has a total of 5 entries
        self.data_store.delete_entry(1234567)
        self.data_store.delete_entry(1234)
        self.data_store.delete_entry(9999)
        # After 3 deletion. 2 left. (2 documents, 1 sentence, 1 group)
        num_doc = len(self.data_store._DataStore__elements["ft.onto.base_ontology.Document"])
        num_sent = len(self.data_store._DataStore__elements["ft.onto.base_ontology.Sentence"])

        self.assertEqual(len(self.data_store._DataStore__entry_dict), 2)
        self.assertEqual(num_doc, 1)
        self.assertEqual(num_sent, 0)

        # delete group
        self.data_store.delete_entry(10123)
        self.assertEqual(len(self.data_store._DataStore__entry_dict), 1)
        self.assertEqual(len(self.data_store._DataStore__elements["ft.onto.base_ontology.Phrase"]), 0)

    def test_delete_entry_nonexist(self):
        # Entry tid does not exist; should raise a KeyError
        with self.assertRaises(KeyError):
            self.data_store.delete_entry(1000)

    def test_delete_entry_by_loc(self):
        self.data_store._delete_entry_by_loc("ft.onto.base_ontology.Document", 1)
        # dict entry is not deleted; only delete entry in element list
        self.assertEqual(len(self.data_store._DataStore__entry_dict), 5)
        self.assertEqual(len(self.data_store._DataStore__elements["ft.onto.base_ontology.Document"]), 1)

        # index_id out of range
        with self.assertRaises(IndexError):
            self.data_store._delete_entry_by_loc("ft.onto.base_ontology.Document", 1)

        # type_name does not exist
        with self.assertRaises(KeyError):
            self.data_store._delete_entry_by_loc("ft.onto.base_ontology.EntityMention", 1)

    def test_is_annotation(self):
        test_type_name = "ft.onto.base_ontology.Sentence"
        is_annot = self.data_store._is_annotation(test_type_name)
        self.assertEqual(is_annot, True)

        test_type_name = "ft.onto.base_ontology.Dependency"
        is_annot = self.data_store._is_annotation(test_type_name)
        self.assertEqual(is_annot, False)

    def test_next_entry(self):
        # next_ent = self.next_entry(1234)
        # self.assertEqual(
        #     next_ent,
        #     [
        #         10,
        #         25,
        #         3456,
        #         "ft.onto.base_ontology.Document",
        #         "Doc class A",
        #         "Negative",
        #         "Class B",
        #     ],
        # )
        # prev_ent = self.prev_entry(3456)
        # self.assertEqual(
        #     prev_ent,
        #     [
        #         0,
        #         5,
        #         1234,
        #         "ft.onto.base_ontology.Document",
        #         None,
        #         "Postive",
        #         None,
        #     ],
        # )
        pass

    def test_get_entry_attribute_by_class(self):
        entry_name_attributes_dict = {
            "data_store_test.TokenTest": [
                "pos",
                "ud_xpos",
                "lemma",
                "chunk",
                "ner",
                "sense",
                "is_root",
                "ud_features",
                "ud_misc",
            ],
            "data_store_test.TitleTest": [],
            "data_store_test.SingleMetricTest": [
                "metric_name",
                "value",
            ],
        }
        for entry_name in entry_name_attributes_dict.keys():
            attribute_result = self.data_store._get_entry_attributes_by_class(
                entry_name
            )
            self.assertEqual(
                attribute_result, entry_name_attributes_dict[entry_name]
            )


if __name__ == "__main__":
    unittest.main()
