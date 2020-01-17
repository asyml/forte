"""This module tests LowerCaser processor."""
import unittest

from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.lowercaser_processor import LowerCaserProcessor


class TestLowerCaserProcessor(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline()
        self.nlp.set_reader(StringReader())
        self.nlp.add_processor(LowerCaserProcessor())
        self.nlp.initialize()

    def test_lowercaser_processor(self):
        document = "This tool is called Forte. The goal of this project to " \
                   "help you build NLP pipelines. NLP has never been made " \
                   "this easy before."
        pack = self.nlp.process(document)
        print(pack)
        print(pack.text)
        assert pack.text == "this tool is called forte. the goal of this " \
                            "project to help you build nlp pipelines. nlp " \
                            "has never been made this easy before."
