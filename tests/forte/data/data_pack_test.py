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
Unit tests for data pack related operations.
"""
import os
import logging
import unittest
from typing import List, Tuple

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.pipeline import Pipeline
from forte.utils import utils
from ft.onto.base_ontology import (
    Token,
    Sentence,
    Document,
    EntityMention,
    PredicateArgument,
    PredicateLink,
    PredicateMention,
    CoreferenceGroup,
    Phrase,
)
from forte.data.readers import OntonotesReader

logging.basicConfig(level=logging.DEBUG)


class DataPackTest(unittest.TestCase):
    def setUp(self) -> None:
        file_dir_path = os.path.dirname(__file__)
        data_path = os.path.abspath(
            os.path.join(
                file_dir_path, "../../../", "data_samples", "ontonotes/one_file"
            )
        )

        pipeline: Pipeline = Pipeline()
        pipeline.set_reader(OntonotesReader())
        pipeline.initialize()
        self.data_pack: DataPack = pipeline.process_one(data_path)

    def test_get_data(self):
        requests = {
            Sentence: ["speaker"],
            Token: ["pos", "sense"],
            EntityMention: [],
            PredicateMention: [],
            PredicateArgument: {"fields": [], "unit": "Token"},
            PredicateLink: {
                "component": utils.get_full_module_name(OntonotesReader),
                "fields": ["parent", "child", "arg_type"],
            },
        }

        # case 1: get sentence context from the beginning
        instances = list(self.data_pack.get_data(Sentence))
        self.assertEqual(len(instances), 2)
        self.assertEqual(
            instances[1]["offset"], len(instances[0]["context"]) + 1
        )

        # case 2: get sentence context from the second instance
        instances = list(self.data_pack.get_data(Sentence, skip_k=1))
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["offset"], 165)

        # case 3: get document context
        instances = list(self.data_pack.get_data(Document, skip_k=0))
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["offset"], 0)

        # case 3.1: test get single
        document: Document = self.data_pack.get_single(Document)
        self.assertEqual(document.text, instances[0]["context"])

        # case 4: test offset out of index
        instances = list(self.data_pack.get_data(Sentence, skip_k=10))
        self.assertEqual(len(instances), 0)

        # case 5: get entries
        instances = list(
            self.data_pack.get_data(Sentence, request=requests, skip_k=1)
        )
        self.assertEqual(len(instances[0].keys()), 9)
        self.assertEqual(len(instances[0]["PredicateLink"]), 4)
        self.assertEqual(len(instances[0]["Token"]), 5)
        self.assertEqual(len(instances[0]["EntityMention"]), 3)

    def test_get_entries(self):
        # case 1: test get annotation
        sent_texts: List[Tuple[int, str]] = []
        for doc in self.data_pack.get(Document):
            for sent in self.data_pack.get(Sentence, doc):
                sent_texts.append(sent.text)

        self.assertEqual(
            sent_texts,
            [
                "The Indonesian billionaire James Riady has agreed "
                "to pay $ 8.5 million and plead guilty to illegally "
                "donating money for Bill Clinton 's 1992 presidential "
                "campaign .",
                "He admits he was trying to influence American policy on "
                "China .",
            ],
        )

        # case 2: test get link
        links: List[Tuple[str, str, str]] = []
        for doc in self.data_pack.get(Document):
            link: PredicateLink
            for link in self.data_pack.get(PredicateLink, doc):
                links.append(
                    (
                        link.get_parent().text,
                        link.get_child().text,
                        link.arg_type,
                    )
                )

        self.assertEqual(
            links,
            [
                ("agreed", "The Indonesian billionaire James Riady", "ARG0"),
                (
                    "agreed",
                    "to pay $ 8.5 million and plead guilty to illegally "
                    "donating money for Bill Clinton 's 1992 presidential campaign",
                    "ARG1",
                ),
                ("pay", "The Indonesian billionaire James Riady", "ARG0"),
                ("pay", "$ 8.5 million", "ARG1"),
                ("plead", "The Indonesian billionaire James Riady", "ARG0"),
                ("plead", "guilty", "ARG1"),
                (
                    "plead",
                    "to illegally donating money for Bill Clinton 's "
                    "1992 presidential campaign",
                    "ARG2",
                ),
                ("donating", "illegally", "ARGM-MNR"),
                ("donating", "money", "ARG1"),
                (
                    "donating",
                    "for Bill Clinton 's 1992 presidential campaign",
                    "ARG2",
                ),
                ("admits", "He", "ARG0"),
                (
                    "admits",
                    "he was trying to influence American policy on China",
                    "ARG1",
                ),
                ("trying", "he", "ARG0"),
                ("trying", "to influence American policy on China", "ARG1"),
                ("influence", "he", "ARG0"),
                ("influence", "American policy on China", "ARG1"),
            ],
        )

        # Case 3: Get Child entries when fetching parents with include_sub_type
        all_anno: List[Tuple[int, str]] = []
        for doc in self.data_pack.get(Document):
            for anno in self.data_pack.get(Annotation, doc):
                # Just to make the test case concise,
                # We will only consider Sentence, EntityMention
                # and PredicateArgument
                if any(
                    isinstance(anno, check)
                    for check in [Sentence, EntityMention, PredicateArgument]
                ):
                    all_anno.append(
                        [type(anno).__name__, anno.text, anno.begin, anno.end]
                    )

        self.assertEqual(
            all_anno,
            [
                [
                    "EntityMention",
                    "The Indonesian billionaire James Riady",
                    0,
                    38,
                ],
                [
                    "PredicateArgument",
                    "The Indonesian billionaire James Riady",
                    0,
                    38,
                ],
                [
                    "PredicateArgument",
                    "The Indonesian billionaire James Riady",
                    0,
                    38,
                ],
                [
                    "PredicateArgument",
                    "The Indonesian billionaire James Riady",
                    0,
                    38,
                ],
                [
                    "Sentence",
                    "The Indonesian billionaire James Riady has agreed to pay $ 8.5 million "
                    "and plead guilty to illegally donating money for Bill Clinton 's 1992 presidential "
                    "campaign .",
                    0,
                    164,
                ],
                ["EntityMention", "Indonesian", 4, 14],
                ["EntityMention", "James Riady", 27, 38],
                [
                    "PredicateArgument",
                    "to pay $ 8.5 million and plead guilty to illegally donating money "
                    "for Bill Clinton 's 1992 presidential campaign",
                    50,
                    162,
                ],
                ["EntityMention", "$ 8.5 million", 57, 70],
                ["PredicateArgument", "$ 8.5 million", 57, 70],
                ["PredicateArgument", "guilty", 81, 87],
                [
                    "PredicateArgument",
                    "to illegally donating money for Bill Clinton 's 1992 "
                    "presidential campaign",
                    88,
                    162,
                ],
                ["PredicateArgument", "illegally", 91, 100],
                ["PredicateArgument", "money", 110, 115],
                [
                    "PredicateArgument",
                    "for Bill Clinton 's 1992 presidential campaign",
                    116,
                    162,
                ],
                ["EntityMention", "Bill Clinton 's", 120, 135],
                ["EntityMention", "1992", 136, 140],
                ["EntityMention", "He", 165, 167],
                ["PredicateArgument", "He", 165, 167],
                [
                    "Sentence",
                    "He admits he was trying to influence American policy on China .",
                    165,
                    228,
                ],
                ["EntityMention", "he", 175, 177],
                ["PredicateArgument", "he", 175, 177],
                ["PredicateArgument", "he", 175, 177],
                [
                    "PredicateArgument",
                    "he was trying to influence American policy on China",
                    175,
                    226,
                ],
                [
                    "PredicateArgument",
                    "to influence American policy on China",
                    189,
                    226,
                ],
                ["EntityMention", "American", 202, 210],
                ["PredicateArgument", "American policy on China", 202, 226],
                ["EntityMention", "China", 221, 226],
            ],
        )

        # test get groups
        groups: List[List[str]] = []
        groups_: List[List[str]] = []
        for doc in self.data_pack.get(Document):
            members: List[str] = []
            group: CoreferenceGroup
            for group in self.data_pack.get(CoreferenceGroup, doc):
                em: EntityMention
                for em in group.get_members():
                    members.append(em.text)
            groups.append(sorted(members))

        # get from string.
        for doc in self.data_pack.get("ft.onto.base_ontology.Document"):
            members: List[str] = []
            groups_: CoreferenceGroup
            for group in self.data_pack.get(
                "ft.onto.base_ontology.CoreferenceGroup", doc
            ):
                em: EntityMention
                for em in group.get_members():
                    members.append(em.text)
            groups_.append(sorted(members))

        self.assertEqual(
            groups, [["He", "The Indonesian billionaire James Riady", "he"]]
        )
        self.assertEqual(groups, groups_)

        # The class cannot be found, so raise value error.
        with self.assertRaises(ValueError):
            for doc in self.data_pack.get("this.is.not.a.valid.class"):
                print(doc)

        # The class can be found, but it is not an Entry, so raise value
        # error.
        with self.assertRaises(ValueError):
            for doc in self.data_pack.get("forte.data.data_pack.DataPack"):
                print(doc)

    def test_delete_entry(self):
        # test delete entry
        sentences = list(self.data_pack.get(Sentence))
        num_sent = len(sentences)
        first_sent = sentences[0]
        self.data_pack.delete_entry(first_sent)
        self.assertEqual(
            len(list(self.data_pack.get_data(Sentence))), num_sent - 1
        )


if __name__ == "__main__":
    unittest.main()
