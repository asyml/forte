"""
Test cases to ensure native Forte code can be imported
with only backbone Forte library installed (without any
extra imports).
"""
import os
import unittest


class ForteImportTest(unittest.TestCase):
    def test_basic_import(self):
        from forte import Pipeline
        from forte.data import DataPack

    def test_import_data(self):
        from forte.data.readers import TerminalReader
        from forte.datasets.mrc.squad_reader import SquadReader

    def test_import_processors(self):
        from forte.processors.writers import PackIdJsonPackWriter
        from forte.processors.third_party import MicrosoftBingTranslator
        from forte.processors.nlp import ElizaProcessor
        from forte.processors.misc import AnnotationRemover
        from forte.processors.base import BaseProcessor
