"""This module tests notebook handling_structued_data.ipynb ."""
import os
import unittest

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from ft.onto.base_ontology import (
    Token,
    Sentence,
    Document,
)
from forte.data.readers import OntonotesReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline


class TestHandlingStructuedData(unittest.TestCase):
    def setUp(self):
        self.root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )
        self.data_path = os.path.abspath(
            os.path.join("data_samples", "ontonotes/one_file")
        )
        pipeline: Pipeline = Pipeline()
        pipeline.set_reader(OntonotesReader())
        pipeline.initialize()
        self.data_pack: DataPack = pipeline.process_one(self.data_path)
        self.doc_text = "The Indonesian billionaire James Riady has agreed to pay $ 8.5 million and plead guilty to illegally donating money for Bill Clinton 's 1992 presidential campaign . He admits he was trying to influence American policy on China ."
        self.sents = [
            "The Indonesian billionaire James Riady has agreed to pay $ 8.5 million and plead guilty to illegally donating money for Bill Clinton 's 1992 presidential campaign .",
            "He admits he was trying to influence American policy on China .",
        ]

    def test_get(self):

        for doc_idx, instance in enumerate(self.data_pack.get(Document)):
            print(doc_idx, "document instance:  ", instance)
            print(doc_idx, "document text:  ", instance.text)
            self.assertEqual(type(instance), Document)
            self.assertTrue(instance.text, self.doc_text)

    def test_get_data(self):
        for doc_idx, doc_d in enumerate(
            self.data_pack.get_data(context_type=Document)
        ):
            print(doc_idx, ":  ", doc_d["context"])
            self.assertTrue(doc_d["context"], self.doc_text)

        data_generator = self.data_pack.get_data(context_type=Sentence)
        for sent_idx, sent_d in enumerate(data_generator):
            self.assertEqual(self.sents[sent_idx], sent_d["context"])
        data_generator = self.data_pack.get_data(
            context_type=Sentence, skip_k=1
        )
        for sent_idx, sent_d in enumerate(data_generator):
            self.assertEqual(self.sents[sent_idx + 1], sent_d["context"])

        requests = {
            Token: ["pos"],
        }

        data_generator = self.data_pack.get_data(
            context_type=Sentence, request=requests, skip_k=1
        )
        num_tokens = 12
        poss = [
            "PRP",
            "VBZ",
            "PRP",
            "VBD",
            "VBG",
            "TO",
            "VB",
            "JJ",
            "NN",
            "IN",
            "NNP",
            ".",
        ]
        tokens = [
            "He",
            "admits",
            "he",
            "was",
            "trying",
            "to",
            "influence",
            "American",
            "policy",
            "on",
            "China",
            ".",
        ]
        for sent_idx, sent_d in enumerate(data_generator):
            self.assertEqual(num_tokens, len(sent_d["Token"]["text"]))
            self.assertEqual(num_tokens, len(sent_d["Token"]["pos"]))
            for p, _p in zip(poss, sent_d["Token"]["pos"]):
                self.assertEqual(p, _p)

            for token_idx, token_txt in enumerate(sent_d["Token"]["text"]):
                self.assertEqual(tokens[token_idx], token_txt)
