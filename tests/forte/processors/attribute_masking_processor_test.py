"""This module tests Attribute Masking processor."""
import os
import unittest

from forte.data.data_pack import DataPack
from forte.data.readers import ConllUDReader
from forte.pipeline import Pipeline
from forte.processors.misc import AttributeMasker
from ft.onto.base_ontology import Token


class TestAttributeMaskingProcessor(unittest.TestCase):
    def setUp(self):
        self.root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )

    def test_without_attribute_masker(self):
        pl = Pipeline[DataPack]()
        pl.set_reader(ConllUDReader())
        pl.initialize()

        for pack in pl.process_dataset(
            os.path.join(self.root_path, "data_samples/conll_ud/")
        ):
            entries = pack.get(Token)
            for entry in entries:
                self.assertIsNotNone(entry.pos)

    def test_attribute_masker(self):
        pl = Pipeline[DataPack]()
        pl.set_reader(ConllUDReader())

        # TODO: should not use class in config.
        config = {"kwargs": {Token: ["pos"]}}

        pl.add(component=AttributeMasker(), config=config)
        pl.initialize()

        for pack in pl.process_dataset(
            os.path.join(self.root_path, "data_samples/conll_ud/")
        ):
            entries = pack.get(Token)
            for entry in entries:
                self.assertIsNone(entry.pos)


if __name__ == "__main__":
    unittest.main()
