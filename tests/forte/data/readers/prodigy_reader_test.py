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
Unit test for ProdigyReader.
"""
import json
import os
import tempfile
import unittest

from forte.data.data_pack import DataPack
from ft.onto.base_ontology import Token, Document, EntityMention
from forte.data.readers import ProdigyReader
from forte.pipeline import Pipeline


class ProdigyReaderTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.fp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                              delete=False)
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(ProdigyReader())
        self.nlp.initialize()
        self.create_sample_file()

    def tearDown(self):
        os.system("rm {}".format(self.fp.name))

    def create_sample_file(self):
        prodigy_entry = {
            "text": "Lorem ipsum dolor sit amet",
            "tokens": [{"text": "Lorem", "start": 0, "end": 5, "id": 0},
                       {"text": "ipsum", "start": 6, "end": 11, "id": 1},
                       {"text": "dolor", "start": 12, "end": 17, "id": 2},
                       {"text": "sit", "start": 18, "end": 21, "id": 3},
                       {"text": "amet", "start": 22, "end": 26, "id": 4}],
            "spans": [{"start": 0, "end": 5, "token_start": 0,
                       "token_end": 1, "label": "sample_latin"},
                      {"start": 12, "end": 26, "token_start": 2,
                       "token_end": 18, "label": "sample_latin"}],
            "meta": {"id": "doc_1", "sect_id": 1, "version": "1"},
            "_input_hash": 123456789,
            "_task_hash": -123456789,
            "_session_id": "abcd", "_view_id": "ner_manual", "answer": "accept"
        }

        # for entry in JSON_file:
        json.dump(prodigy_entry, self.fp)
        self.fp.write('\n')
        json.dump(prodigy_entry, self.fp)
        self.fp.write('\n')
        self.fp.close()

    def test_packs(self):
        doc_exists = False
        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.fp.name):
            # get documents from pack
            for doc in pack.get_entries(Document):
                doc_exists = True
                self.token_check(doc, pack)
                self.label_check(doc, pack)
            self.assertEqual(pack.meta.doc_id, "doc_1")
        self.assertTrue(doc_exists)

    def token_check(self, doc, pack):
        doc_text = doc.text
        # Compare document text with tokens
        tokens = [token.text for token in
                  pack.get_entries(Token, doc)]
        self.assertEqual(tokens[2], "dolor")
        self.assertEqual(doc_text.replace(" ", ""), "".join(tokens))

    def label_check(self, doc, pack):
        # make sure that the labels are read in correctly
        labels = [label.ner_type for label in
                  pack.get_entries(EntityMention, doc)]
        self.assertEqual(labels, ["sample_latin", "sample_latin"])


if __name__ == '__main__':
    unittest.main()
