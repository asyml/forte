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
Unit tests for Reader.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from ft.onto.base_ontology import Token, Sentence, Document, EntityMention
from forte.data.readers import (
    OntonotesReader, ProdigyReader, CoNLL03Reader, StringReader)
from forte.pipeline import Pipeline
from forte.processors.base.tests.dummy_pack_processor import DummyPackProcessor


class OntonotesReaderPipelineTest(unittest.TestCase):
    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/ontonotes/00"

        self.nlp = Pipeline()

        self.nlp.set_reader(OntonotesReader())
        self.nlp.add_processor(DummyPackProcessor())

        self.nlp.initialize()

    def test_process_next(self):
        doc_exists = False
        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get sentence from pack
            for sentence in pack.get_entries(Sentence):
                doc_exists = True
                sent_text = sentence.text
                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get_entries(Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))
        self.assertTrue(doc_exists)


class CoNLL03ReaderPipelineTest(unittest.TestCase):
    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "data_samples/conll03"

        self.nlp = Pipeline()

        self.nlp.set_reader(CoNLL03Reader())
        self.nlp.add_processor(DummyPackProcessor())
        self.nlp.add_processor(DummyPackProcessor())

        self.nlp.initialize()

    def test_process_next(self):
        doc_exists = False
        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get sentence from pack
            for sentence in pack.get_entries(Sentence):
                doc_exists = True
                sent_text = sentence.text
                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get_entries(Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))
        self.assertTrue(doc_exists)


class ProdigyReaderTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.fp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                              delete=False)
        self.nlp = Pipeline()
        self.nlp.set_reader(ProdigyReader())
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


class StringReaderPipelineTest(unittest.TestCase):
    def setUp(self):
        # Define and config the Pipeline
        self.dataset_path = "forte/data/readers/tests/"

        self.pl1 = Pipeline()
        self._cache_directory = Path(os.path.join(os.getcwd(), "cache_data"))
        self.pl1.set_reader(StringReader())

        self.pl2 = Pipeline()
        self.pl2.set_reader(StringReader())

        self.text = (
            "The plain green Norway spruce is displayed in the gallery's "
            "foyer. Wentworth worked as an assistant to sculptor Henry Moore "
            "in the late 1960s. His reputation as a sculptor grew in the "
            "1980s.")

    def test_reader(self):
        self._process()
        self._read_caching()

    def _process(self):
        doc_exists = False
        for pack in self.pl1.process_dataset([self.text]):
            doc_exists = True
            self.assertEqual(self.text, pack.text)
        self.assertTrue(doc_exists)

    def _read_caching(self):
        doc_exists = False
        # get processed pack from dataset
        for pack in self.pl2.process_dataset([self.text]):
            doc_exists = True
            self.assertEqual(self.text, pack.text)
        self.assertTrue(doc_exists)

    def tearDown(self):
        os.system("rm -r {}".format(self._cache_directory))


if __name__ == '__main__':
    unittest.main()
