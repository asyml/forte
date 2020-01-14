"""This module tests spaCy processors."""
import unittest
from ddt import ddt, data

from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.spacy_processors import SpacyProcessor
from ft.onto.base_ontology import Token


@ddt
class TestSpacyProcessor(unittest.TestCase):
    def setUp(self):
        self.spacy = Pipeline()
        self.spacy.set_reader(StringReader())

        config = HParams({
            "processors": "tokenize",
            "lang": "en_core_web_sm",
            # Language code for the language to build the Pipeline
            "use_gpu": False
        }, SpacyProcessor.default_hparams())
        self.spacy.add_processor(SpacyProcessor(), config=config)
        self.spacy.initialize()

    def test_spacy_processor(self):
        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = self.spacy.process(document)

        # Check document
        self.assertEqual(pack.text, document)

        # Check tokens
        tokens = [x.text for x in pack.annotations if isinstance(x, Token)]
        document = document.replace('.', ' .')
        self.assertEqual(tokens, document.split())

    @data(
        "tokenize",
        "tokenize, pos",
        "tokenize, pos, lemma",
        "tokenize, lemma",
        "lemma",
        "lemma, pos"
    )
    def test_spacy_variation_pipeline(self, value):
        spacy = Pipeline()
        spacy.set_reader(StringReader())

        config = HParams({
            "processors": value,
            "lang": "en_core_web_sm",
            # Language code for the language to build the Pipeline
            "use_gpu": False
        }, SpacyProcessor.default_hparams())
        spacy.add_processor(SpacyProcessor(), config=config)
        spacy.initialize()

        sentences = ["This tool is called Forte.",
                     "The goal of this project to help you build NLP "
                     "pipelines.",
                     "NLP has never been made this easy before."]
        document = ' '.join(sentences)
        pack = spacy.process(document)

        if "tokenize" in value:
            tokens = [x.text for x in pack.annotations if isinstance(x, Token)]
            pos = [x.pos for x in pack.annotations if isinstance(x, Token)]
            lemma = [x.lemma for x in pack.annotations if isinstance(x, Token)]
            document = document.replace('.', ' .')
            self.assertEqual(tokens, document.split())

            if "pos" in value:
                self.assertIsNotNone(pos)
            else:
                none_pos = [None] * len(pos)
                self.assertListEqual(pos, none_pos)

            if "lemma" in value:
                self.assertIsNotNone(lemma)
            else:
                none_lemma = [None] * len(lemma)
                self.assertListEqual(lemma, none_lemma)
        else:
            tokens = [x for x in pack.annotations if isinstance(x, Token)]
            self.assertListEqual(tokens, [])


if __name__ == "__main__":
    unittest.main()
