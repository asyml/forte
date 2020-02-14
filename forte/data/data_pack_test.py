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
import logging
import unittest

from ft.onto.base_ontology import (
    Token, Sentence, Document, EntityMention, PredicateArgument, PredicateLink,
    PredicateMention)
from forte.data.readers import OntonotesReader

logging.basicConfig(level=logging.DEBUG)


class DataPackTest(unittest.TestCase):

    def setUp(self) -> None:
        self.reader = OntonotesReader()
        data_path = "data_samples/ontonotes/00/abc_0059.gold_conll"
        self.data_pack = list(self.reader.parse_pack(data_path))[0]

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
                "component": self.reader.component_name,
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

        # case 4: test offset out of index
        instances = list(self.data_pack.get_data(Sentence, skip_k=10))
        self.assertEqual(len(instances), 0)

        # case 5: get entries
        instances = list(self.data_pack.get_data(Sentence,
                                                 request=requests,
                                                 skip_k=1))
        self.assertEqual(len(instances[0].keys()), 9)
        self.assertEqual(len(instances[0]["PredicateLink"]), 4)
        self.assertEqual(len(instances[0]["Token"]), 5)
        self.assertEqual(len(instances[0]["EntityMention"]), 3)

        # case 6: test delete entry
        num_sent = len([self.data_pack.get(Sentence)])
        first_sent = [self.data_pack.get(Sentence)][0]
        self.data_pack.delete_entry(first_sent)
        self.assertEqual(len([self.data_pack.get_data(Sentence)]), num_sent - 1)


if __name__ == '__main__':
    unittest.main()
