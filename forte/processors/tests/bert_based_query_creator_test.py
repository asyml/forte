"""This module tests Query Creator processor."""
import unittest
import os
import tempfile
import shutil

from ddt import ddt, data, unpack
from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.processors import BertBasedQueryCreator
from forte.data.readers import MultiPackSentenceReader
from forte.data.ontology import Query
from ft.onto.base_ontology import Token, Sentence


@ddt
class TestBertBasedQueryCreator(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @data((["Hello, good morning",
            "This is a tool for NLP"],))
    @unpack
    def test_pipeline(self, texts):
        for idx, text in enumerate(texts):
            file_path = os.path.join(self.test_dir, f"{idx+1}.txt")
            with open(file_path, 'w') as f:
                f.write(text)

        nlp = Pipeline()
        reader_config = HParams({"input_pack_name": "query",
                                 "output_pack_name": "output"},
                                MultiPackSentenceReader.default_hparams())
        nlp.set_reader(reader=MultiPackSentenceReader(), config=reader_config)
        config = HParams({"model": {"name": "bert-base-uncased"},
                          "tokenizer": {"name": "bert-base-uncased"},
                          "max_seq_length": 128,
                          "query_pack_name": "query"}, None)
        nlp.add_processor(BertBasedQueryCreator(), config=config)

        nlp.initialize()

        for idx, m_pack in enumerate(nlp.process_dataset(self.test_dir)):
            query_pack = m_pack.get_pack("query")
            self.assertEqual(len(query_pack.generics), 1)
            self.assertIsInstance(query_pack.generics[0], Query)
            query = query_pack.generics[0].value
            self.assertEqual(query.shape, (1, 768))
