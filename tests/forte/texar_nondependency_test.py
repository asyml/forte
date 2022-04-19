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
        from forte.data import DataStore
        from forte.data import SinglePackSelector
        from forte.data import BaseIndex

        from forte.train_pipeline import TrainPipeline

    def test_import_data(self):
        from forte.data.readers import TerminalReader
        from forte.datasets.mrc.squad_reader import SquadReader

    def test_import_processors(self):
        from forte.processors.writers import PackIdJsonPackWriter
        from forte.processors.third_party import MicrosoftBingTranslator
        from forte.processors.nlp import ElizaProcessor
        from forte.processors.misc import AnnotationRemover
        from forte.processors.base import BaseProcessor
        from forte.processors.data_augment import BaseDataAugmentProcessor
        from forte.processors.ir.search_processor import SearchProcessor
        from forte.processors.nlp import CoNLLNERPredictor
        from forte.processors.third_party import (
            MicrosoftBingTranslator,
            MachineTranslationProcessor,
        )

    def test_import_evaluator(self):
        from forte.evaluation.ner_evaluator import CoNLLNEREvaluator
        from forte.evaluation.base import Evaluator

    def test_import_models(self):
        from forte.models.ner.conditional_random_field import (
            ConditionalRandomField,
        )
        from forte.models.srl import SRLSpan, Span, RawExample, Example

    def test_import_trainer(self):
        from forte.trainer.base import BaseTrainer
        from forte.trainer.ner_trainer import CoNLLNERTrainer
