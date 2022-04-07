"""
Unit tests for dependency test.
"""
from importlib import import_module
import os
import unittest


class DependencyTest(unittest.TestCase):
    def setUp(self):
        self.dataset_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                *([os.path.pardir] * 4),
                "data_samples/squad_v2.0/dev-v2.0-sample.json"
            )
        )

    def test_import_data(self):
        # Read with no replacements
        import_module("forte.data")
        import_module("forte.data.converter")
        import_module("forte.data.extractors")
        import_module("forte.data.vocabulary")
        import_module("forte.data.readers")

    def test_import_datasets(self):
        import_module("forte.datasets")

    def test_import_models(self):
        import_module("forte.models")
        import_module("forte.models.da_rl")
        import_module("forte.models.ner")
        import_module("forte.models.srl")
        import_module("forte.models.srl_new")

    def test_import_processors(self):
        import_module("forte.processors.data_augment.algorithms")
        import_module("forte.processors.ir")
        import_module("forte.processors.misc")
        import_module("forte.processors.nlp")
        import_module("forte.processors.third_party")
