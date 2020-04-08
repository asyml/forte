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

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.utils import utils
from ft.onto.base_ontology import (
    Token, Sentence, Document, EntityMention, PredicateArgument, PredicateLink,
    PredicateMention)
from forte.data.readers import OntonotesReader

logging.basicConfig(level=logging.DEBUG)


class DataPackTest(unittest.TestCase):

    def setUp(self) -> None:
        file_dir_path = os.path.dirname(__file__)
        data_path = os.path.join(file_dir_path, os.pardir, os.pardir,
                                 'test_data', 'ontonotes')

        pipeline = Pipeline()
        pipeline.set_reader(OntonotesReader())
        pipeline.initialize()
        self.data_pack: DataPack = pipeline.process_one(data_path)

    def test_get_data(self):
        requests = {
            Sentence: ["speaker"],
            Token: ["pos", "sense"],
            EntityMention: [],
            PredicateMention: [],
            PredicateArgument: {
                "fields": [],
                "unit": "Token"
            },
            PredicateLink: {
                "component": utils.get_full_module_name(OntonotesReader),
                "fields": ["parent", "child", "arg_type"]
            }
        }

        # case 1: get sentence context from the beginning
        instances = list(self.data_pack.get_data(Sentence))
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[1]["offset"],
                         len(instances[0]["context"]) + 1)

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
        self.assertEqual(document.text, instances[0]['context'])

        # case 4: test offset out of index
        instances = list(self.data_pack.get_data(Sentence, skip_k=10))
        self.assertEqual(len(instances), 0)

        # case 5: get entries
        instances = list(self.data_pack.get_data(
            Sentence, request=requests, skip_k=1))
        self.assertEqual(len(instances[0].keys()), 9)
        self.assertEqual(len(instances[0]["PredicateLink"]), 4)
        self.assertEqual(len(instances[0]["Token"]), 5)
        self.assertEqual(len(instances[0]["EntityMention"]), 3)

        # case 6: test delete entry
        sentences = list(self.data_pack.get(Sentence))
        num_sent = len(sentences)
        first_sent = sentences[0]
        self.data_pack.delete_entry(first_sent)
        self.assertEqual(len(list(self.data_pack.get_data(Sentence))),
                         num_sent - 1)


if __name__ == '__main__':
    unittest.main()
