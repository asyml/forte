"""
Unit tests for Pipeline.
"""
import os
import unittest
import json
import tempfile
from forte.pipeline import Pipeline
from forte.data.readers import OntonotesReader, ProdigyReader, CoNLL03Reader
from forte.processors.dummy_pack_processor import DummyPackProcessor
from forte.data.ontology import relation_ontology, base_ontology, conll03_ontology
from forte.data.ontology.relation_ontology import *


class OntonotesReaderPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        # Define and config the Pipeline
        self.dataset_path = "examples/ontonotes_sample_dataset/00"

        self.nlp = Pipeline()
        self.nlp.set_ontology(relation_ontology)

        self.nlp.set_reader(OntonotesReader())
        self.processor = DummyPackProcessor()
        self.nlp.add_processor(self.processor)

    def test_process_next(self):

        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get sentence from pack
            for sentence in pack.get_entries(Sentence):
                sent_text = sentence.text
                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get_entries(Token, sentence)]
                self.assertEqual(sent_text, " ".join(tokens))


class CoNLLReaderPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        # Define and config the Pipeline
        self.dataset_path = "examples/"

        self.nlp = Pipeline()
        self.nlp.set_ontology(conll03_ontology)

        self.nlp.set_reader(CoNLL03Reader())
        self.processor = DummyPackProcessor()
        self.nlp.add_processor(self.processor)

    def test_process_next(self):

        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.dataset_path):
            # get sentence from pack
            for doc in pack.get_entries(conll03_ontology.Document):
                doc_text = doc.text
                # second method to get entry in a sentence
                tokens = [token.text for token in
                          pack.get_entries(conll03_ontology.Token, doc)]
                self.assertEqual(doc_text.strip(), " ".join(tokens))


class ProdigyReaderTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.fp = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        self.nlp = Pipeline()
        self.nlp.set_ontology(base_ontology)
        self.nlp.set_reader(ProdigyReader())
        self.processor = DummyPackProcessor()
        self.nlp.add_processor(self.processor)
        self.create_sample_file()

    def tearDown(self):
        os.system("rm {}".format(self.fp.name))

    def create_sample_file(self) -> None:

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
        # get processed pack from dataset
        for pack in self.nlp.process_dataset(self.fp.name):
            # get documents from pack
            for doc in pack.get_entries(base_ontology.Document):
                self.token_check(doc, pack)
                self.label_check(doc, pack)

    def token_check(self, doc, pack):
        doc_text = doc.text
        # Compare document text with tokens
        tokens = [token.text for token in
                  pack.get_entries(base_ontology.Token, doc)]
        self.assertEqual(tokens[2], "dolor")
        self.assertEqual(doc_text.replace(" ", ""), "".join(tokens))

    def label_check(self, doc, pack):
        # make sure that the labels are read in correctly
        labels = [label.ner_type for label in
                  pack.get_entries(base_ontology.EntityMention, doc)]
        self.assertEqual(labels, ["sample_latin", "sample_latin"])


if __name__ == '__main__':
    unittest.main()
