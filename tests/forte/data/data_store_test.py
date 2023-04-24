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

import os
import logging
import unittest
import copy
from forte.data.ontology.core import Entry, FList

from ft.onto.base_ontology import Document, Sentence
from sortedcontainers import SortedList
from typing import Any, List, Optional, Dict, Tuple, Union
from dataclasses import dataclass
from forte.data.data_store import DataStore
from forte.data.ontology.top import (
    Annotation,
    Generics,
)
from forte.data.data_pack import DataPack
from forte.common import constants
from forte.data.ontology.core import FDict
from ft.onto.base_ontology import Classification


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
        self.reference_type_attributes = {
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
                "parent_class": set(),
            },
            "ft.onto.base_ontology.Sentence": {
                "attributes": {
                    "begin": {"index": 2, "type": (type(None), (int,))},
                    "end": {"index": 3, "type": (type(None), (int,))},
                    "payload_idx": {"index": 4, "type": (type(None), (int,))},
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
                "parent_class": set(),
            },
            "forte.data.ontology.top.Annotation": {
                "attributes": {
                    "begin": {"index": 2, "type": (type(None), (int,))},
                    "end": {"index": 3, "type": (type(None), (int,))},
                    "payload_idx": {"index": 4, "type": (type(None), (int,))},
                },
                "parent_class": {"Entry"},
            },
        }

        self.base_type_attributes = {
            "forte.data.ontology.top.Generics": {"parent_class": {"Entry"}},
            "forte.data.ontology.top.Link": {
                "attributes": {
                    "parent_type": {"type": (type(None), (str,)), "index": 2},
                    "child_type": {"type": (type(None), (str,)), "index": 3},
                    "parent": {"type": (Union, (int, type(None))), "index": 4},
                    "child": {"type": (Union, (int, type(None))), "index": 5},
                },
                "parent_class": {"BaseLink"},
            },
            "forte.data.ontology.top.Group": {
                "attributes": {
                    "members": {"type": (list, (int,)), "index": 2},
                    "member_type": {"type": (type(None), (str,)), "index": 3},
                },
                "parent_class": {"Entry", "BaseGroup"},
            },
            "forte.data.ontology.top.MultiPackGeneric": {
                "parent_class": {"Entry", "MultiEntry"}
            },
            "forte.data.ontology.top.MultiPackLink": {
                "attributes": {
                    "parent_type": {"type": (type(None), (str,)), "index": 2},
                    "child_type": {"type": (type(None), (str,)), "index": 3},
                    "parent": {"type": (tuple, (Tuple,)), "index": 4},
                    "child": {"type": (tuple, (Tuple,)), "index": 5},
                },
                "parent_class": {"MultiEntry", "BaseLink"},
            },
            "forte.data.ontology.top.MultiPackGroup": {
                "attributes": {
                    "members": {"type": (list, (Tuple, int, int)), "index": 2},
                    "member_type": {"type": (type(None), (str,)), "index": 3},
                },
                "parent_class": {"Entry", "MultiEntry", "BaseGroup"},
            },
            "forte.data.ontology.top.Query": {"parent_class": {"Generics"}},
            "forte.data.ontology.top.AudioAnnotation": {
                "attributes": {
                    "begin": {"index": 2, "type": (type(None), (int,))},
                    "end": {"index": 3, "type": (type(None), (int,))},
                    "payload_idx": {"index": 4, "type": (type(None), (int,))},
                },
                "parent_class": {"Entry"},
            },
        }

        DataStore._type_attributes["ft.onto.base_ontology.Document"] = {
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
            "parent_class": set(),
        }

        DataStore._type_attributes["ft.onto.base_ontology.Sentence"] = {
            "attributes": {
                "begin": {"index": 2, "type": (type(None), (int,))},
                "end": {"index": 3, "type": (type(None), (int,))},
                "payload_idx": {"index": 4, "type": (type(None), (int,))},
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
            "parent_class": set(),
        }

        DataStore._type_attributes["forte.data.ontology.top.Annotation"] = {
            "attributes": {
                "begin": {"index": 2, "type": (type(None), (int,))},
                "end": {"index": 3, "type": (type(None), (int,))},
                "payload_idx": {"index": 4, "type": (type(None), (int,))},
            },
            "parent_class": {"Entry"},
        }
        # The order is [Document, Sentence]. Initialize 2 entries in each list.
        # Document entries have tid 1234, 3456.
        # Sentence entries have tid 9999, 1234567.
        # The type id for Document is 0, Sentence is 1.

        ref1 = [
            1234,
            "ft.onto.base_ontology.Document",
            0,
            5,
            0,
            [],
            {"Positive": 0},
            {},
        ]
        ref2 = [
            3456,
            "ft.onto.base_ontology.Document",
            10,
            25,
            1,
            ["Doc class A"],
            {"Negative": 0},
            {},
        ]
        ref3 = [
            9999,
            "ft.onto.base_ontology.Sentence",
            6,
            9,
            0,
            "teacher",
            1,
            {"Positive": 0},
            {},
            {},
        ]
        ref4 = [
            1234567,
            "ft.onto.base_ontology.Sentence",
            55,
            70,
            0,
            None,
            None,
            {"Negative": 0},
            {"Class C": 0},
            {},
        ]
        ref5 = [7654, "forte.data.ontology.top.Annotation", 10, 20, 0]

        sorting_fn = lambda s: (
            s[
                self.reference_type_attributes[
                    "forte.data.ontology.top.Annotation"
                ]["attributes"]["begin"]["index"]
            ],
            s[
                self.reference_type_attributes[
                    "forte.data.ontology.top.Annotation"
                ]["attributes"]["end"]["index"]
            ],
        )
        self.data_store._DataStore__elements = {
            "ft.onto.base_ontology.Document": SortedList(
                [ref1, ref2], key=sorting_fn
            ),
            "ft.onto.base_ontology.Sentence": SortedList(
                [ref3, ref4], key=sorting_fn
            ),
            # empty list corresponds to Entry, test only
            "forte.data.ontology.core.Entry": SortedList([]),
            "forte.data.ontology.top.Annotation": SortedList(
                [ref5], key=sorting_fn
            ),
            "forte.data.ontology.top.Group": [
                [
                    10123,
                    "forte.data.ontology.top.Group",
                    [9999, 1234567],
                    "ft.onto.base_ontology.Sentence",
                ],
                [
                    23456,
                    "forte.data.ontology.top.Group",
                    [1234, 3456],
                    "ft.onto.base_ontology.Document",
                ],
                [
                    34567,
                    "forte.data.ontology.top.Group",
                    [1234, 7654],
                    "forte.data.ontology.top.Annotation",
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
            1234: ref1,
            3456: ref2,
            9999: ref3,
            1234567: ref4,
            7654: ref5,
        }
        self.data_store._DataStore__tid_idx_dict = {
            10123: ["forte.data.ontology.top.Group", 0],
            23456: ["forte.data.ontology.top.Group", 1],
            34567: ["forte.data.ontology.top.Group", 2],
            88888: ["forte.data.ontology.top.Link", 0],
        }

    def test_get_type_info(self):
        # initialize
        empty_data_store = DataStore()
        DataStore._type_attributes = {}
        # test get new type
        doc_attr_dict = empty_data_store._get_type_info(
            "ft.onto.base_ontology.Document"
        )
        empty_data_store._get_type_info("ft.onto.base_ontology.Sentence")
        self.assertEqual(len(empty_data_store._DataStore__elements), 0)

        self.assertEqual(
            DataStore._type_attributes["ft.onto.base_ontology.Sentence"],
            self.reference_type_attributes["ft.onto.base_ontology.Sentence"],
        )
        self.assertEqual(
            DataStore._type_attributes["ft.onto.base_ontology.Document"],
            self.reference_type_attributes["ft.onto.base_ontology.Document"],
        )
        # test the return value
        self.assertEqual(
            doc_attr_dict,
            DataStore._type_attributes["ft.onto.base_ontology.Document"],
        )

        # test get invalid type
        with self.assertRaisesRegex(
            ValueError, "Class not found in invalid.Type"
        ):
            empty_data_store._get_type_info("invalid.Type")
        self.assertTrue("invalid.Type" not in DataStore._type_attributes)

        # test get existing type
        doc_attr_dict = empty_data_store._get_type_info(
            "ft.onto.base_ontology.Document"
        )
        self.assertEqual(len(DataStore._type_attributes), 2)
        self.assertEqual(
            doc_attr_dict,
            DataStore._type_attributes["ft.onto.base_ontology.Document"],
        )

        # test get type info with ontology file input
        with self.assertRaisesRegex(
            RuntimeError,
            "DataStore is initialized with no existing types. Setting"
            "dynamically_add_type to False without providing onto_file_path"
            "will lead to no usable type in DataStore.",
        ):
            DataStore(dynamically_add_type=False)

        DataStore._type_attributes = self.reference_type_attributes
        # TODO: need more tests for ontology file input

    def test_entry_methods(self):
        sent_type = "ft.onto.base_ontology.Sentence"
        doc_type = "ft.onto.base_ontology.Document"
        ann_type = "forte.data.ontology.top.Annotation"
        group_type = "forte.data.ontology.top.Group"
        sent_list = list(self.data_store._DataStore__elements[sent_type])
        doc_list = list(self.data_store._DataStore__elements[doc_type])
        ann_list = list(
            self.data_store.co_iterator_annotation_like(
                list(self.data_store._get_all_subclass(ann_type, True))
            )
        )

        group_list = list(self.data_store._DataStore__elements[group_type])
        sent_entries = list(self.data_store.all_entries(sent_type))
        doc_entries = list(self.data_store.all_entries(doc_type))
        ann_entries = list(self.data_store.all_entries(ann_type))

        self.assertEqual(sent_list, sent_entries)
        self.assertEqual(doc_list, doc_entries)
        self.assertEqual(ann_list, ann_entries)

        self.assertEqual(self.data_store.num_entries(sent_type), len(sent_list))
        self.assertEqual(self.data_store.num_entries(doc_type), len(doc_list))
        self.assertEqual(
            self.data_store.num_entries(ann_type), len(ann_entries)
        )

        # remove two sentence
        self.data_store.delete_entry(9999)
        self.data_store.delete_entry(1234567)
        self.assertEqual(
            self.data_store.num_entries(sent_type), len(sent_list) - 2
        )
        self.assertEqual(
            self.data_store.num_entries(ann_type), len(ann_list) - 2
        )  # test parent entry count
        # add a sentence back and count
        add_count = 5
        for i in range(add_count):
            self.data_store.add_entry_raw(
                type_name="ft.onto.base_ontology.Sentence",
                attribute_data=[i, i + 1],
            )
        self.assertEqual(
            self.data_store.num_entries(sent_type),
            len(sent_list) - 2 + add_count,
        )
        self.assertEqual(
            self.data_store.num_entries(ann_type), len(ann_list) - 2 + add_count
        )  # test parent entry count

        # remove a group
        self.data_store.delete_entry(23456)
        num_group_entries = self.data_store.num_entries(group_type)
        self.assertEqual(num_group_entries, len(group_list) - 1)

    def test_co_iterator_annotation_like(self):
        type_names = [
            "ft.onto.base_ontology.Sentence",
            "ft.onto.base_ontology.Document",
        ]

        # test sort by begin index
        ordered_elements = [
            [
                1234,
                "ft.onto.base_ontology.Document",
                0,
                5,
                0,
                [],
                {"Positive": 0},
                {},
            ],
            [
                9999,
                "ft.onto.base_ontology.Sentence",
                6,
                9,
                0,
                "teacher",
                1,
                {"Positive": 0},
                {},
                {},
            ],
            [
                3456,
                "ft.onto.base_ontology.Document",
                10,
                25,
                1,
                ["Doc class A"],
                {"Negative": 0},
                {},
            ],
            [
                1234567,
                "ft.onto.base_ontology.Sentence",
                55,
                70,
                0,
                None,
                None,
                {"Negative": 0},
                {"Class C": 0},
                {},
            ],
        ]

        elements = list(self.data_store.co_iterator_annotation_like(type_names))
        self.assertEqual(elements, ordered_elements)

        # test sort by end index
        ordered_elements = [
            [
                1234,
                "ft.onto.base_ontology.Document",
                0,
                5,
                0,
                [],
                {"Positive": 0},
                {},
            ],
            [
                3456,
                "ft.onto.base_ontology.Document",
                0,
                25,
                1,
                ["Doc class A"],
                {"Negative": 0},
                {},
            ],
            [
                9999,
                "ft.onto.base_ontology.Sentence",
                6,
                9,
                0,
                "teacher",
                1,
                {"Positive": 0},
                {},
                {},
            ],
            [
                1234567,
                "ft.onto.base_ontology.Sentence",
                55,
                70,
                0,
                None,
                None,
                {"Negative": 0},
                {"Class C": 0},
                {},
            ],
        ]
        doc_tn = "ft.onto.base_ontology.Document"
        sent_tn = "ft.onto.base_ontology.Sentence"

        begin_idx = self.data_store._type_attributes[doc_tn][
            constants.ATTR_INFO_KEY
        ]["begin"][constants.ATTR_INDEX_KEY]

        self.data_store._DataStore__elements[doc_tn][0][begin_idx] = 0
        self.data_store._DataStore__elements[doc_tn][1][begin_idx] = 0
        elements = list(self.data_store.co_iterator_annotation_like(type_names))
        self.assertEqual(elements, ordered_elements)

        # test sort by input type_names
        ordered_elements1 = [
            [
                9999,
                "ft.onto.base_ontology.Sentence",
                0,
                5,
                0,
                "teacher",
                1,
                {"Positive": 0},
                {},
                {},
            ],
            [
                1234,
                "ft.onto.base_ontology.Document",
                0,
                5,
                0,
                [],
                {"Positive": 0},
                {},
            ],
            [
                3456,
                "ft.onto.base_ontology.Document",
                0,
                25,
                1,
                ["Doc class A"],
                {"Negative": 0},
                {},
            ],
            [
                1234567,
                "ft.onto.base_ontology.Sentence",
                55,
                70,
                0,
                None,
                None,
                {"Negative": 0},
                {"Class C": 0},
                {},
            ],
        ]
        ordered_elements2 = copy.deepcopy(ordered_elements1)
        ordered_elements2[0] = ordered_elements1[1]
        ordered_elements2[1] = ordered_elements1[0]

        begin_idx = self.data_store._type_attributes[sent_tn][
            constants.ATTR_INFO_KEY
        ]["begin"][constants.ATTR_INDEX_KEY]
        end_idx = self.data_store._type_attributes[sent_tn][
            constants.ATTR_INFO_KEY
        ]["end"][constants.ATTR_INDEX_KEY]

        self.data_store._DataStore__elements[sent_tn][0][begin_idx] = 0
        self.data_store._DataStore__elements[sent_tn][0][end_idx] = 5
        elements = list(self.data_store.co_iterator_annotation_like(type_names))
        self.assertEqual(elements, ordered_elements1)
        type_names.reverse()
        elements = list(self.data_store.co_iterator_annotation_like(type_names))
        self.assertEqual(elements, ordered_elements2)

        token_tn = "ft.onto.base_ontology.Token"
        # include Token to test non-exist list

        def value_err_fn():
            type_names.append(token_tn)
            list(self.data_store.co_iterator_annotation_like(type_names))

        self.assertRaises(ValueError, value_err_fn)

        # test iterate empty list
        def value_err_fn():
            type_names = [token_tn]
            list(self.data_store.co_iterator_annotation_like(type_names))

        self.assertRaises(ValueError, value_err_fn)

    def test_add_annotation_raw(self):
        # test add Document entry
        tid_doc: int = self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.Document",
            attribute_data=[1, 5],
        )

        # test add Sentence entry
        tid_sent: int = self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.Sentence",
            attribute_data=[5, 8],
        )
        num_doc = self.data_store.get_length("ft.onto.base_ontology.Document")

        num_sent = self.data_store.get_length("ft.onto.base_ontology.Sentence")

        self.assertEqual(num_doc, 3)
        self.assertEqual(num_sent, 3)
        self.assertEqual(len(self.data_store._DataStore__tid_ref_dict), 7)
        self.assertEqual(
            self.data_store.get_entry(tid=tid_doc)[0],
            [tid_doc, "ft.onto.base_ontology.Document", 1, 5, 0, [], {}, {}],
        )
        self.assertEqual(
            self.data_store.get_entry(tid=tid_sent)[0],
            [
                tid_sent,
                "ft.onto.base_ontology.Sentence",
                5,
                8,
                0,
                None,
                None,
                {},
                {},
                {},
            ],
        )

        # test add new annotation type
        tid_em: int = self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.EntityMention",
            attribute_data=[10, 12],
        )
        num_phrase = self.data_store.get_length(
            "ft.onto.base_ontology.EntityMention"
        )

        self.assertEqual(num_phrase, 1)
        self.assertEqual(len(self.data_store._DataStore__tid_ref_dict), 8)
        self.assertEqual(
            self.data_store.get_entry(tid=tid_em)[0],
            [tid_em, "ft.onto.base_ontology.EntityMention", 10, 12, 0, None],
        )

        # test add duplicate Sentence entry
        tid_sent_duplicate: int = self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.Sentence",
            allow_duplicate=False,
            attribute_data=[5, 8],
        )
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "ft.onto.base_ontology.Sentence"
                ]
            ),
            num_sent,
        )
        self.assertEqual(tid_sent, tid_sent_duplicate)
        self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.Sentence",
            allow_duplicate=False,
            attribute_data=[5, 9],
        )
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "ft.onto.base_ontology.Sentence"
                ]
            ),
            num_sent + 1,
        )

        # check add annotation raw with tid
        tid = 77
        self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.Sentence",
            tid=tid,
            attribute_data=[0, 1],
        )
        self.assertEqual(
            self.data_store.get_entry(tid=77)[0],
            [
                tid,
                "ft.onto.base_ontology.Sentence",
                0,
                1,
                0,
                None,
                None,
                {},
                {},
                {},
            ],
        )

    def test_add_audio_annotation_raw(self):
        # test add Document entry
        tid_recording: int = self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.Recording",
            attribute_data=[1, 5],
        )
        # test add Sentence entry
        tid_audio_utterance: int = self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.AudioUtterance",
            attribute_data=[5, 8],
        )
        tid_utterance: int = self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.Utterance",
            attribute_data=[5, 8],
        )
        # check number of Recording
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "ft.onto.base_ontology.Recording"
                ]
            ),
            1,
        )
        # check number of AudioUtterance
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "ft.onto.base_ontology.AudioUtterance"
                ]
            ),
            1,
        )
        # check number of Utterance
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "ft.onto.base_ontology.Utterance"
                ]
            ),
            1,
        )
        tid = 77
        self.data_store.add_entry_raw(
            type_name="ft.onto.base_ontology.Recording",
            tid=tid,
            attribute_data=[0, 1],
        )
        self.assertEqual(
            self.data_store.get_entry(tid=77)[0],
            [tid, "ft.onto.base_ontology.Recording", 0, 1, 0, []],
        )

    def test_add_generics_raw(self):
        # test add Document entry
        tid_generics: int = self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.Generics", attribute_data={}
        )
        # check number of Generics
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "forte.data.ontology.top.Generics"
                ]
            ),
            1,
        )
        tid = 77
        self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.Generics",
            tid=tid,
        )
        self.assertEqual(
            self.data_store.get_entry(tid=77)[0],
            [tid, "forte.data.ontology.top.Generics"],
        )

    def test_add_link_raw(self):
        self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.Link",
            attribute_data=[
                "ft.onto.base_ontology.Sentence",
                "ft.onto.base_ontology.Document",
            ],
        )
        # check number of Link
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "forte.data.ontology.top.Link"
                ]
            ),
            2,
        )

        # check add link with tid
        tid = 77
        self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.Link",
            attribute_data=[
                "ft.onto.base_ontology.Sentence",
                "ft.onto.base_ontology.Document",
            ],
            tid=tid,
        )
        self.assertEqual(
            self.data_store.get_entry(tid=77)[0],
            [
                tid,
                "forte.data.ontology.top.Link",
                "ft.onto.base_ontology.Sentence",
                "ft.onto.base_ontology.Document",
                None,
                None,
            ],
        )

    def test_add_group_raw(self):
        self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.Group",
            attribute_data=["ft.onto.base_ontology.Document"],
        )
        # check number of Group
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "forte.data.ontology.top.Group"
                ]
            ),
            4,
        )

        # check add group with tid
        tid = 77
        self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.Group",
            attribute_data=["ft.onto.base_ontology.Sentence"],
            tid=tid,
        )
        self.assertEqual(
            self.data_store.get_entry(tid=77)[0],
            [
                tid,
                "forte.data.ontology.top.Group",
                [],
                "ft.onto.base_ontology.Sentence",
            ],
        )

    def test_add_multientry_raw(self):
        self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.MultiPackGeneric",
        )
        # check number of MultiPackGeneric
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "forte.data.ontology.top.MultiPackGeneric"
                ]
            ),
            1,
        )

        self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.MultiPackGroup",
            attribute_data=["ft.onto.base_ontology.Sentence"],
        )
        # check number of MultiPackGeneric
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "forte.data.ontology.top.MultiPackGroup"
                ]
            ),
            1,
        )

        self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.MultiPackLink",
            attribute_data=[
                "ft.onto.base_ontology.Sentence",
                "ft.onto.base_ontology.Document",
            ],
        )
        # check number of MultiPackGeneric
        self.assertEqual(
            len(
                self.data_store._DataStore__elements[
                    "forte.data.ontology.top.MultiPackLink"
                ]
            ),
            1,
        )

        # check add MultiPackLink with tid
        tid = 77
        self.data_store.add_entry_raw(
            type_name="forte.data.ontology.top.MultiPackLink",
            attribute_data=[
                "ft.onto.base_ontology.Sentence",
                "ft.onto.base_ontology.Document",
            ],
            tid=tid,
        )
        self.assertEqual(
            self.data_store.get_entry(tid=tid)[0],
            [
                tid,
                "forte.data.ontology.top.MultiPackLink",
                "ft.onto.base_ontology.Sentence",
                "ft.onto.base_ontology.Document",
                None,
                None,
            ],
        )

    def test_get_attribute(self):
        speaker = self.data_store.get_attribute(9999, "speaker")
        classifications = self.data_store.get_attribute(3456, "classifications")

        self.assertEqual(speaker, "teacher")
        self.assertEqual(classifications, {})

        # Entry with such tid does not exist
        with self.assertRaisesRegex(KeyError, "Entry with tid 1111 not found."):
            self.data_store.get_attribute(1111, "speaker")

        # Get attribute field that does not exist
        with self.assertRaisesRegex(
            KeyError, "ft.onto.base_ontology.Sentence has no class attribute."
        ):
            self.data_store.get_attribute(9999, "class")

    def test_get_attributes_of_tid(self):
        result_dict = self.data_store.get_attributes_of_tid(
            9999, ["begin", "end", "speaker"]
        )
        result_dict2 = self.data_store.get_attributes_of_tid(
            3456, ["payload_idx", "classifications"]
        )

        self.assertEqual(result_dict["begin"], 6)
        self.assertEqual(result_dict["end"], 9)
        self.assertEqual(result_dict["speaker"], "teacher")
        self.assertEqual(result_dict2["payload_idx"], 1)
        self.assertEqual(result_dict2["classifications"], {})

        # Entry with such tid does not exist
        with self.assertRaisesRegex(KeyError, "Entry with tid 1111 not found."):
            self.data_store.get_attributes_of_tid(1111, ["speaker"])

        # Get attribute field that does not exist
        with self.assertRaisesRegex(
            KeyError, "ft.onto.base_ontology.Sentence has no class attribute."
        ):
            self.data_store.get_attributes_of_tid(9999, ["class"])

    def test_get_attributes_of_tids(self):
        tids_attrs: list[dict]
        # tids_attrs2: list[dict]
        tids_attrs = self.data_store.get_attributes_of_tids(
            [9999, 3456], ["begin", "end", "payload_idx"]
        )
        tids_attrs2 = self.data_store.get_attributes_of_tids(
            [9999], ["begin", "speaker"]
        )

        self.assertEqual(tids_attrs2[0]["begin"], 6)
        self.assertEqual(tids_attrs[0]["end"], 9)
        self.assertEqual(tids_attrs[1]["payload_idx"], 1)
        self.assertEqual(tids_attrs2[0]["speaker"], "teacher")

        # Entry with such tid does not exist
        with self.assertRaisesRegex(KeyError, "Entry with tid 1111 not found."):
            self.data_store.get_attributes_of_tids([1111], ["speaker"])

        # Get attribute field that does not exist
        with self.assertRaisesRegex(
            KeyError, "ft.onto.base_ontology.Sentence has no class attribute."
        ):
            self.data_store.get_attributes_of_tids([9999], ["class"])

    def test_get_attributes_of_type(self):
        # get document entries
        instances = list(
            self.data_store.get_attributes_of_type(
                "ft.onto.base_ontology.Document",
                ["begin", "end", "payload_idx"],
            )
        )
        # print(instances)
        self.assertEqual(len(instances), 2)
        # check tid
        self.assertEqual(instances[0]["tid"], 1234)
        self.assertEqual(instances[0]["end"], 5)
        self.assertEqual(instances[1]["tid"], 3456)
        self.assertEqual(instances[1]["begin"], 10)

        # For types other than annotation, group or link, not support include_subtype
        instances = list(
            self.data_store.get_attributes_of_type(
                "forte.data.ontology.core.Entry", ["begin", "end"]
            )
        )
        self.assertEqual(len(instances), 0)

        self.assertEqual(
            self.data_store.get_length("forte.data.ontology.core.Entry"), 0
        )

        # get annotations with subclasses and range annotation
        instances = list(
            self.data_store.get_attributes_of_type(
                "forte.data.ontology.top.Annotation",
                ["begin", "end"],
                range_span=(1, 20),
            )
        )
        self.assertEqual(len(instances), 2)

        # get groups with subclasses
        # instances = list(self.data_store.get_attributes_of_type(
        #     "forte.data.ontology.top.Group", ["begin", "end"]))
        # self.assertEqual(len(instances), 3)

        # # get groups with subclasses and range annotation
        # instances = list(
        #     self.data_store.get(
        #         "forte.data.ontology.top.Group", range_span=(1, 20)
        #     )
        # )
        # self.assertEqual(len(instances), 0)
        #
        # # get links with subclasses
        # instances = list(self.data_store.get("forte.data.ontology.top.Link"))
        # self.assertEqual(len(instances), 1)
        #
        # # get links with subclasses and range annotation
        # instances = list(
        #     self.data_store.get(
        #         "forte.data.ontology.top.Link", range_span=(0, 9)
        #     )
        # )
        # self.assertEqual(len(instances), 1)
        #
        # # get links with subclasses and range annotation
        # instances = list(
        #     self.data_store.get(
        #         "forte.data.ontology.top.Link", range_span=(4, 11)
        #     )
        # )
        # self.assertEqual(len(instances), 0)

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
        sent = self.data_store.get_entry(1234567)
        self.assertEqual(
            sent,
            (
                [
                    1234567,
                    "ft.onto.base_ontology.Sentence",
                    55,
                    70,
                    0,
                    None,
                    None,
                    {"Negative": 0},
                    {"Class C": 0},
                    {},
                ],
                "ft.onto.base_ontology.Sentence",
            ),
        )

        # Entry with such tid does not exist
        with self.assertRaises(KeyError):
            for doc in self.data_store.get_entry(1111):
                print(doc)

    def test_get_entry_index(self):
        self.assertEqual(self.data_store.get_entry_index(1234567), 1)

        # Entry with such tid does not exist
        with self.assertRaises(KeyError):
            self.data_store.get_entry_index(1111)

    def test_get(self):
        # get document entries
        instances = list(self.data_store.get("ft.onto.base_ontology.Document"))
        # print(instances)
        self.assertEqual(len(instances), 2)
        # check tid
        self.assertEqual(instances[0][0], 1234)
        self.assertEqual(instances[1][0], 3456)

        # For types other than annotation, group or link, not support include_subtype
        instances = list(self.data_store.get("forte.data.ontology.core.Entry"))
        self.assertEqual(len(instances), 0)

        self.assertEqual(
            self.data_store.get_length("forte.data.ontology.core.Entry"), 0
        )

        # get annotations with subclasses and range annotation
        instances = list(
            self.data_store.get(
                "forte.data.ontology.top.Annotation", range_span=(1, 20)
            )
        )
        self.assertEqual(len(instances), 2)

        # get groups with subclasses
        instances = list(self.data_store.get("forte.data.ontology.top.Group"))
        self.assertEqual(len(instances), 3)

        # get groups with subclasses and range annotation
        instances = list(
            self.data_store.get(
                "forte.data.ontology.top.Group", range_span=(1, 20)
            )
        )
        self.assertEqual(len(instances), 0)

        # get links with subclasses
        instances = list(self.data_store.get("forte.data.ontology.top.Link"))
        self.assertEqual(len(instances), 1)

        # get links with subclasses and range annotation
        instances = list(
            self.data_store.get(
                "forte.data.ontology.top.Link", range_span=(0, 9)
            )
        )
        self.assertEqual(len(instances), 1)

        # get links with subclasses and range annotation
        instances = list(
            self.data_store.get(
                "forte.data.ontology.top.Link", range_span=(4, 11)
            )
        )
        self.assertEqual(len(instances), 0)

    def test_delete_entry(self):
        # delete annotation
        # has a total of 5 entries
        self.assertEqual(len(self.data_store._DataStore__tid_ref_dict), 5)
        self.data_store.delete_entry(1234567)
        self.data_store.delete_entry(1234)
        self.data_store.delete_entry(9999)
        # After 3 deletion. 5 left. (1 document, 1 annotation, 2 groups, 1 link)
        num_doc = self.data_store.get_length("ft.onto.base_ontology.Document")
        num_group = self.data_store.get_length("forte.data.ontology.top.Group")

        self.assertEqual(len(self.data_store._DataStore__tid_ref_dict), 2)
        self.assertEqual(num_doc, 1)
        self.assertEqual(num_group, 3)

        # delete group
        self.data_store.delete_entry(10123)
        self.assertEqual(len(self.data_store._DataStore__tid_idx_dict), 3)
        self.data_store.delete_entry(23456)
        self.data_store.delete_entry(34567)
        self.assertTrue(
            "forte.data.ontology.top.Group"
            not in self.data_store._DataStore__elements
        )

        # delete link
        self.assertTrue(
            "forte.data.ontology.top.Link"
            in self.data_store._DataStore__elements
        )

        self.data_store.delete_entry(88888)
        self.assertTrue(
            "forte.data.ontology.top.Link"
            not in self.data_store._DataStore__elements
        )

    def test_delete_entry_nonexist(self):
        # Entry tid does not exist; should raise a KeyError
        with self.assertRaises(KeyError):
            self.data_store.delete_entry(1000)

    def test_delete_entry_by_loc(self):
        self.data_store._delete_entry_by_loc(
            "ft.onto.base_ontology.Document", 1
        )
        # dict entry is not deleted; only delete entry in element list
        self.assertEqual(len(self.data_store._DataStore__tid_ref_dict), 5)
        self.assertEqual(
            self.data_store.get_length("ft.onto.base_ontology.Document"),
            1,
        )

        # index_id out of range
        with self.assertRaises(IndexError):
            self.data_store._delete_entry_by_loc(
                "ft.onto.base_ontology.Document", 1
            )

        # type_name does not exist
        with self.assertRaises(KeyError):
            self.data_store._delete_entry_by_loc(
                "ft.onto.base_ontology.EntityMention", 1
            )

    def test_is_annotation(self):
        test_type_name = "ft.onto.base_ontology.Sentence"
        is_annot = self.data_store._is_annotation(test_type_name)
        self.assertEqual(is_annot, True)

        test_type_name = "ft.onto.base_ontology.Dependency"
        is_annot = self.data_store._is_annotation(test_type_name)
        self.assertEqual(is_annot, False)

    def test_next_entry(self):
        next_ent = self.data_store.next_entry(1234)
        self.assertEqual(
            next_ent,
            [
                3456,
                "ft.onto.base_ontology.Document",
                10,
                25,
                1,
                ["Doc class A"],
                {"Negative": 0},
                {},
            ],
        )
        # Last entry in list does not have a next entry.
        self.assertIsNone(self.data_store.next_entry(3456))
        # Raise exception when tid does not exist
        with self.assertRaises(KeyError):
            self.data_store.next_entry(1111)

        prev_ent = self.data_store.prev_entry(3456)
        self.assertEqual(
            prev_ent,
            [
                1234,
                "ft.onto.base_ontology.Document",
                0,
                5,
                0,
                [],
                {"Positive": 0},
                {},
            ],
        )
        # First entry in list does not have a previous entry.
        self.assertIsNone(self.data_store.prev_entry(1234))
        # Raise exception when tid does not exist
        with self.assertRaises(KeyError):
            self.data_store.prev_entry(1111)

        # test next/prev with delete on group entries
        self.assertIsNone(self.data_store.prev_entry(10123))
        self.assertIsNone(self.data_store.next_entry(34567))
        self.assertIsNone(self.data_store.prev_entry(88888))
        self.assertIsNone(self.data_store.next_entry(88888))

        self.data_store.delete_entry(23456)
        next_ent = self.data_store.next_entry(10123)
        self.assertEqual(
            next_ent,
            [
                34567,
                "forte.data.ontology.top.Group",
                [1234, 7654],
                "forte.data.ontology.top.Annotation",
            ],
        )

        prev_ent = self.data_store.prev_entry(34567)
        self.assertEqual(
            prev_ent,
            [
                10123,
                "forte.data.ontology.top.Group",
                [9999, 1234567],
                "ft.onto.base_ontology.Sentence",
            ],
        )

        self.data_store.delete_entry(34567)
        self.assertIsNone(self.data_store.next_entry(10123))

    def test_get_entry_attribute_by_class(self):
        entry_name_attributes_dict = {
            "data_store_test.TokenTest": [
                "begin",
                "end",
                "payload_idx",
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
            "data_store_test.TitleTest": ["begin", "end", "payload_idx"],
            "data_store_test.SingleMetricTest": [
                "metric_name",
                "value",
            ],
        }
        for entry_name in entry_name_attributes_dict.keys():
            attribute_result = list(
                self.data_store._get_entry_attributes_by_class(entry_name)
            )
            self.assertEqual(
                attribute_result, entry_name_attributes_dict[entry_name]
            )

    def test_is_subclass(self):
        import forte

        self.assertEqual(
            DataStore._type_attributes["ft.onto.base_ontology.Document"][
                "parent_class"
            ],
            set(),
        )

        self.assertTrue(
            self.data_store._is_subclass(
                "ft.onto.base_ontology.Document",
                forte.data.ontology.top.Annotation,
            )
        )

        self.assertTrue(
            self.data_store._is_subclass(
                "ft.onto.base_ontology.Document", forte.data.ontology.core.Entry
            )
        )

        self.assertFalse(
            self.data_store._is_subclass(
                "ft.onto.base_ontology.Document", forte.data.ontology.top.Link
            )
        )

        self.assertEqual(
            DataStore._type_attributes["ft.onto.base_ontology.Document"][
                "parent_class"
            ],
            {
                "forte.data.ontology.top.Annotation",
                "forte.data.ontology.core.Entry",
            },
        )

        self.assertFalse(
            self.data_store._is_subclass(
                "ft.onto.base_ontology.Title",
                forte.data.ontology.top.Annotation,
                no_dynamic_subclass=True,
            )
        )

        DataStore._type_attributes["ft.onto.base_ontology.Title"][
            "parent_class"
        ].add("forte.data.ontology.top.Annotation")

        self.assertTrue(
            self.data_store._is_subclass(
                "ft.onto.base_ontology.Title",
                forte.data.ontology.top.Annotation,
                no_dynamic_subclass=True,
            )
        )
        DataStore._type_attributes["ft.onto.base_ontology.Title"][
            "parent_class"
        ].add("forte.data.ontology.top.Link")
        self.assertTrue(
            self.data_store._is_subclass(
                "ft.onto.base_ontology.Title", forte.data.ontology.top.Link
            )
        )

    def test_check_onto_file(self):
        expected_type_attributes = {
            "ft.onto.test.Description": {
                "attributes": {
                    "begin": {"index": 2, "type": (type(None), (int,))},
                    "end": {"index": 3, "type": (type(None), (int,))},
                    "payload_idx": {"index": 4, "type": (type(None), (int,))},
                    "author": {"index": 5, "type": (type(None), (str,))},
                    "passage_id": {"index": 6, "type": (type(None), (str,))},
                },
                "parent_class": {"forte.data.ontology.top.Annotation"},
            },
            "ft.onto.test.EntityMention": {
                "attributes": {
                    "begin": {"index": 2, "type": (type(None), (int,))},
                    "end": {"index": 3, "type": (type(None), (int,))},
                    "payload_idx": {"index": 4, "type": (type(None), (int,))},
                    "ner_type": {"index": 5, "type": (type(None), (str,))},
                },
                "parent_class": {"forte.data.ontology.top.Annotation"},
            },
            "ft.onto.test.MedicalEntityMention": {
                "attributes": {
                    "begin": {"index": 2, "type": (type(None), (int,))},
                    "end": {"index": 3, "type": (type(None), (int,))},
                    "payload_idx": {"index": 4, "type": (type(None), (int,))},
                    "ner_type": {"index": 5, "type": (type(None), (str,))},
                    "umls_entities": {"index": 6, "type": (type(None), (int,))},
                    "umls_link": {"index": 7, "type": (type(None), (str,))},
                },
                "parent_class": {"ft.onto.test.EntityMention"},
            },
        }
        data_store_from_file = DataStore(
            onto_file_path=os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "ontology/test_specs/test_check_onto_file.json",
                )
            )
        )
        # Check whether `_type_attributes` contains all items in `expected_type_attributes`
        union_dict: Dict = dict(
            data_store_from_file._type_attributes, **expected_type_attributes
        )
        self.assertDictEqual(data_store_from_file._type_attributes, union_dict)

        # DataStores share a static type_attribute dict.
        data_store_non_file = DataStore()
        self.assertDictEqual(
            data_store_non_file._type_attributes,
            data_store_from_file._type_attributes,
        )

    def test_entry_conversion(self):
        data_pack = DataPack()
        data_pack._data_store = self.data_store
        data_pack.set_text(
            "Forte is a data-centric framework designed to engineer complex ML workflows. Forte allows practitioners to build ML components in a composable and modular way. Behind the scene, it introduces DataPack, a standardized data structure for unstructured data, distilling good software engineering practices such as reusability, extensibility, and flexibility into ML solutions."
        )
        for tid in self.data_store._DataStore__tid_ref_dict:
            entry = data_pack._entry_converter.get_entry_object(
                tid=tid, pack=data_pack
            )
            for attribute in self.data_store._get_entry_attributes_by_class(
                self.data_store.get_entry(tid=tid)[1]
            ):
                entry_val = getattr(entry, attribute)
                ref_val = self.data_store.get_attribute(
                    tid=tid, attr_name=attribute
                )
                if isinstance(ref_val, (list, dict)):
                    continue
                self.assertEqual(entry_val, ref_val)


if __name__ == "__main__":
    unittest.main()
