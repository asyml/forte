"""
Unit tests for dummy processor.
"""
import os
import unittest

from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.dummy_processor import DummyRelationExtractor


class DummyProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.reader = OntonotesReader()
        data_path = os.path.join("examples/abc_0059.gold_conll")
        self.data_pack = self.reader.parse_pack(data_path)

        self.processor = DummyRelationExtractor()
        self.processor.set_input_info()
        self.processor.set_output_info()

    def test_processor(self):
        # case 1: process ner_data
        link_num = len(self.data_pack.links)
        self.processor.process(self.data_pack)
        self.assertEqual(link_num + 11, len(self.data_pack.links))


if __name__ == '__main__':
    unittest.main()
