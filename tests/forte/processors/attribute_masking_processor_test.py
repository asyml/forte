"""This module tests Attribute Masking processor."""
import unittest
import os

from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers import CoNLL03Reader
from forte.processors import AttributeMasker
from ft.onto.base_ontology import Token


class TestAttributeMaskingProcessor(unittest.TestCase):

    def test_without_attribute_masker(self):
        pl = Pipeline()
        pl.set_reader(CoNLL03Reader())
        pl.initialize()

        for pack in pl.process_dataset("data_samples/conll03/"):
            entries = pack.get_entries_by_type(Token)
            for entry in entries:
                self.assertIsNotNone(entry.ner)

    def test_attribute_masker(self):
        pl = Pipeline()
        pl.set_reader(CoNLL03Reader())
        config = {
            "kwargs": {
                Token: ["ner"]
            }
        }

        pl.add_processor(processor=AttributeMasker(), config=config)
        pl.initialize()

        for pack in pl.process_dataset("data_samples/conll03/"):
            entries = pack.get_entries_by_type(Token)
            for entry in entries:
                self.assertIsNone(entry.ner)
