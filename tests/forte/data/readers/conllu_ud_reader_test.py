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
Tests for conllU reader
"""

import os
import unittest

from typing import List

from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, Document, Dependency
from forte.data.readers import ConllUDReader
from forte.data.data_pack import DataPack


class ConllUDReaderTest(unittest.TestCase):
    def setUp(self):
        """
        Reading the data into data_pack object to be used in the tests
        """
        file_dir_path = os.path.dirname(__file__)
        conll_ud_dir = os.path.abspath(os.path.join(file_dir_path,
                                                    *([os.pardir] * 4),
                                                    'data_samples/conll_ud'))
        pl = Pipeline()
        pl.set_reader(ConllUDReader())
        pl.initialize()

        self.data_packs: List[DataPack] = \
            [data_pack for data_pack in pl.process_dataset(conll_ud_dir)]
        self.doc_ids = ["weblog-blogspot.com_nominations_20041117172713_ENG_"
                        "20041117_172713",
                        "weblog-blogspot.com_nominations_20041117172713_ENG_"
                        "20041117_172714"]

    def test_reader_text(self):
        expected_docs_text = [
            ["From the AP comes this story :",
             "President Bush on Tuesday nominated two individuals to "
             "replace retiring jurists on federal courts in the "
             "Washington area ."],
            ["Bush nominated Jennifer M. Anderson for a 15 - year "
             "term as associate judge of the Superior Court of the "
             "District of Columbia , replacing Steffen W. Graae ."]
        ]

        self.assertEqual(len(self.data_packs), 2)

        for doc_index, expected_doc_id in enumerate(self.doc_ids):
            data_pack = self.data_packs[doc_index]
            self.assertTrue(data_pack.meta.doc_id == expected_doc_id)

            doc_entry = None
            for d in data_pack.get(Document):
                doc_entry = d
                break

            expected_doc_text = expected_docs_text[doc_index]
            self.assertEqual(doc_entry.text, ' '.join(expected_doc_text))

            sent_entries = data_pack.get(Sentence)

            for sent_entry, expected_sent_text in zip(
                    sent_entries, expected_doc_text):
                self.assertEqual(sent_entry.text, expected_sent_text)

    def test_reader_dependency_tree(self):
        doc_index = 1
        data_pack = self.data_packs[doc_index]
        expected_doc_id = self.doc_ids[doc_index]
        self.assertTrue(data_pack.meta.doc_id == expected_doc_id)
        self.assertEqual(len(list(data_pack.get_entries(Sentence))), 1)
        dependencies = data_pack.get_entries(Dependency)
        for link in dependencies:
            root_token = get_dependency_tree_root(link, data_pack)
            self.assertEqual(root_token.text, "nominated")


def get_dependency_tree_root(link, data_pack):
    """
    Returns the root token of the dependency tree.

    Args:
        link: The intermediate dependency link.
        data_pack: The data pack to be worked on.

    Returns:

    """
    # TODO: make it robust enough to handle cycles for enhanced dependencies
    token = link.get_parent()
    if token.is_root:
        return token
    parent_link = list(data_pack.get_links_by_child(token))[0]
    return token if token.is_root else get_dependency_tree_root(parent_link,
                                                                data_pack)


if __name__ == "__main__":
    unittest.main()
