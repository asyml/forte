"""This module tests Attribute Masking processor."""
import unittest

from forte.data.data_pack import DataPack
from forte.data.readers import ConllUDReader
from forte.pipeline import Pipeline
from forte.processors import AttributeMasker
from ft.onto.base_ontology import Token


class TestAttributeMaskingProcessor(unittest.TestCase):

    def test_without_attribute_masker(self):
        pl = Pipeline[DataPack]()
        pl.set_reader(ConllUDReader())
        pl.initialize()

        for pack in pl.process_dataset("data_samples/conll_ud/"):
            entries = pack.get(Token)
            for entry in entries:
                self.assertIsNotNone(entry.pos)

    def test_attribute_masker(self):
        pl = Pipeline[DataPack]()
        pl.set_reader(ConllUDReader())
        config = {
            "kwargs": {
                Token: ["pos"]
            }
        }

        pl.add(component=AttributeMasker(), config=config)
        pl.initialize()

        for pack in pl.process_dataset("data_samples/conll_ud/"):
            entries = pack.get(Token)
            for entry in entries:
                self.assertIsNone(entry.pos)
