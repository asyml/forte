from forte.data.readers import ClassificationDatasetReader
from fortex.huggingface import ZeroShotClassifier
from forte.pipeline import Pipeline
from fortex.nltk import NLTKSentenceSegmenter
from ft.onto.base_ontology import Sentence

import os
import shutil
from typing import Dict
from transformers import T5Tokenizer, T5ForConditionalGeneration
from forte import Pipeline
from forte.data import DataPack
from forte.common import Resources, Config
from forte.processors.base import PackProcessor
from forte.data.readers import PlainTextReader
import unittest
from notebook_classes import MachineTranslationProcessor

from forte.pipeline import Pipeline


class TestMTInferencePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline: Pipeline = Pipeline[DataPack]()
        self.pipeline.set_reader(PlainTextReader())
        self.dir_path = os.path.abspath(
            os.path.join("data_samples", "machine_translation")
        )

    def run(self):
        self.pipeline.run(self.dir_path)
