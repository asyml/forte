"""This module tests notebook handling_structued_data.ipynb ."""
import os
import unittest

from forte.data.data_pack import DataPack
from forte.data.readers import ConllUDReader
from forte.pipeline import Pipeline
from forte.processors.misc import AttributeMasker
from ft.onto.base_ontology import Token
import os

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.utils import utils
from ft.onto.base_ontology import (
    Token,
    Sentence,
    Document,
    AudioAnnotation,
    AudioUtterance,
)
from forte.data.ontology import Annotation
from forte.data.readers import OntonotesReader, AudioReader
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

    def test_get(self):

        for doc_idx, instance in enumerate(self.data_pack.get(Document)):
            print(doc_idx, "document instance:  ", instance)
            print(doc_idx, "document text:  ", instance.text)

    def test_get_data(self):
        for doc_idx, doc_d in enumerate(
            self.data_pack.get_data(context_type=Document)
        ):
            print(doc_idx, ":  ", doc_d["context"])

        data_generator = self.data_pack.get_data(context_type=Sentence)
        for sent_idx, sent_d in enumerate(data_generator):
            print(sent_idx, sent_d["context"])

        data_generator = self.data_pack.get_data(
            context_type=Sentence, skip_k=1
        )
        for sent_idx, sent_d in enumerate(data_generator):
            print(sent_idx, sent_d["context"])

        requests = {
            Token: ["pos"],
        }

        data_generator = self.data_pack.get_data(
            context_type=Sentence, request=requests, skip_k=1
        )
        for sent_idx, sent_d in enumerate(data_generator):
            print(sent_idx, sent_d["context"])
            print(sent_d["Token"]["pos"])
            print("Token list length:", len(sent_d["Token"]["text"]))
            print("POS list length:", len(sent_d["Token"]["pos"]))

        data_generator = self.data_pack.get_data(
            context_type=Sentence, request=requests, skip_k=1
        )
        for sent_idx, sent_d in enumerate(data_generator):
            print(sent_idx, sent_d["context"])
            for token_txt, token_pos in zip(
                sent_d["Token"]["text"], sent_d["Token"]["pos"]
            ):
                print(token_txt, token_pos)
