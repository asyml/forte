"""
Unit tests for dummy processor.
"""
import os
import unittest

from nlp.pipeline.data.readers.ontonotes_reader import OntonotesReader
from nlp.pipeline.processors.dummy_processor import DummyRelationExtractor


class DummyProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.reader = OntonotesReader()
        data_path = os.path.join("../../../examples/abc_0059.gold_conll")
        self.data_pack = self.reader.read(data_path)

        self.processor = DummyRelationExtractor()

    def test_processor(self):
        # case 1: process ner_data
        link_num = len(self.data_pack.links)
        self.processor.process(self.data_pack)
        self.assertEqual(link_num + 11, len(self.data_pack.links))


if __name__ == '__main__':
    unittest.main()
