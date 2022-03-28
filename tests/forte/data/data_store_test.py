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
from forte.data.ontology.ontology_code_generator import OntologyCodeGenerator


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
        # The order is [Document, Sentence]. Initialize 2 entries in each list.
        # Document entries have tid 1234, 3456.
        # Sentence entries have tid 9999, 1234567.
        # The type id for Document is 0, Sentence is 1.

        # self.data_store._DataStore__type_index_dict = {
        #     "ft.onto.base_ontology.Document": 0,
        #     "ft.onto.base_ontology.Sentence": 1,
        #     "forte.data.ontology.core.Entry": 2,
        # }

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
        }

    def test_add_annotation_raw(self):
        # # test add Document entry
        # self.data_store.add_annotation_raw(0, 1, 5)
        # # test add Sentence entry
        # self.data_store.add_annotation_raw(1, 5, 8)
        # num_doc = len(self.data_store._DataStore__elements[0])
        # num_sent = len(self.data_store._DataStore__elements[1])

        # self.assertEqual(num_doc, 3)
        # self.assertEqual(num_sent, 3)
        # self.assertEqual(len(self.data_store._DataStore__entry_dict), 6)
        pass

    def test_get_attr(self):
        speaker = self.data_store.get_attribute(9999, "speaker")
        classifications = self.data_store.get_attribute(3456, "classifications")

        self.assertEqual(speaker, "teacher")
        self.assertEqual(classifications, "Class B")

        # Entry with such tid does not exist
        with self.assertRaisesRegex(KeyError, "Entry with tid 1111 not found."):
            self.data_store.get_attribute(1111, "speaker")

        # Get attribute field that does not exist
        with self.assertRaisesRegex(
            ValueError, "ft.onto.base_ontology.Sentence has no class attribute."
        ):
            self.data_store.get_attribute(9999, "class")

    def test_set_attr(self):
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
            ValueError, "ft.onto.base_ontology.Sentence has no speak attribute."
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
        self.assertEqual(len(instances), 4)

        # get entries without subclasses
        instances = list(self.data_store.get("forte.data.ontology.core.Entry", include_sub_type=False))
        self.assertEqual(len(instances), 0)

    def test_delete_entry(self):
        # # In test_add_annotation_raw(), we add 2 entries. So 6 in total.
        # self.data_store.delete_entry(1234567)
        # self.data_store.delete_entry(1234)
        # self.data_store.delete_entry(9999)
        # # After 3 deletion. 3 left. (2 documents and 1 sentence)
        # num_doc = len(self.data_store._DataStore__elements[0])
        # num_sent = len(self.data_store._DataStore__elements[1])

        # self.assertEqual(len(self.data_store._DataStore__entry_dict), 3)
        # self.assertEqual(num_doc, 2)
        # self.assertEqual(num_sent, 1)

        # # Entry with such tid does not exist
        # with self.assertRaises(ValueError):
        #     for doc in self.data_store.delete_entry(1111):
        #         print(doc)
        pass

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

    def test_check_onto_file(self):
        expected_type_attributes = {
            "ftx.onto.clinical.UMLSConceptLink": {
                "attributes": {
                    "cui": 4,
                    "name": 5,
                    "definition": 6,
                    "tuis": 7,
                    "aliases": 8,
                    "score": 9,
                },
                "parent_entry": "forte.data.ontology.top.Generics",
            },
            "ftx.onto.clinical.MedicalEntityMention": {
                "attributes": {
                    "umls_link": 4,
                    "umls_entities": 5,
                },
                "parent_entry": "ft.onto.base_ontology.EntityMention",
            }
        }
        data_store_from_file = DataStore(onto_file_path="forte/ontology_specs/medical.json")
        self.assertDictContainsSubset(expected_type_attributes, data_store_from_file._type_attributes)

        data_store_non_file = DataStore()
        self.assertDictEqual(data_store_non_file._type_attributes, {})
    
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
