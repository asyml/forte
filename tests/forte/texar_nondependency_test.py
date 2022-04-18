"""
Test cases to ensure native Forte code can be imported
with only backbone Forte library installed (without any
extra imports).
Current nondepency packages:
- texar
- torch
"""
import os
import unittest


class ForteImportTest(unittest.TestCase):
    def test_basic_import(self):
        from forte.data import DataPack
        from forte.data import DataStore
        from forte.data import SinglePackSelector
        from forte.data import BaseIndex

    def test_import_data(self):
        from forte.data.readers import TerminalReader
        from forte.datasets.mrc.squad_reader import SquadReader

    def test_import_processors(self):
        from forte.processors.writers import (
            PackIdJsonPackWriter,
        )  # check dependency on torch
        from forte.processors.nlp import (
            ElizaProcessor,
        )  # check dependency on torch
        from forte.processors.misc import (
            AnnotationRemover,
        )  # check dependency on torch
        from forte.processors.base import (
            BaseProcessor,
        )  # check dependency on torch
        from forte.processors.data_augment import (
            BaseDataAugmentProcessor,
        )  # check dependency on torch
        from forte.processors.ir.search_processor import (
            SearchProcessor,
        )  # check dependency on torch

    def test_import_evaluator(self):
        from forte.evaluation.ner_evaluator import (
            CoNLLNEREvaluator,
        )  # check dependency on torch
        from forte.evaluation.base import Evaluator

    def test_import_models(self):
        from forte.models.srl import (
            SRLSpan,
            Span,
            RawExample,
            Example,
        )  # check dependency on torch

        # pass

    def test_import_trainer(self):
        from forte.trainer.base import BaseTrainer  # check dependency on torch
