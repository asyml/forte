"""This module tests Stanford NLP processors."""
import os
import unittest

from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.stanfordnlp_processor import StandfordNLPProcessor
from ft.onto.base_ontology import Token, Sentence


class TestStanfordNLPProcessor(unittest.TestCase):
    def setUp(self):
        self.stanford_nlp = Pipeline()
        self.stanford_nlp.set_reader(StringReader())
        models_path = os.getcwd()
        config = HParams({
            "processors": "tokenize",
            "lang": "en",
            # Language code for the language to build the Pipeline
            "use_gpu": False
        }, StandfordNLPProcessor.default_hparams())
        self.stanford_nlp.add_processor(StandfordNLPProcessor(models_path),
                                        config=config)
        self.stanford_nlp.initialize()

    # TODO
    @unittest.skip("We need to test this without needing to download models "
                   "everytime")
    def test_stanford_processor(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = self.stanford_nlp.process(document)
        print(pack)
