"""This module tests notebook handling_structued_data.ipynb ."""
import os
import unittest
from notebook_classes import MachineTranslationProcessor
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token
import os

from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from ft.onto.base_ontology import (
    Token,
    Sentence,
    Document,
)
from forte.data.readers import OntonotesReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline

# set up
from forte import Pipeline
from forte.data import DataPack
from forte.data.readers import PlainTextReader
import os


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.root_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir,
                os.pardir,
                os.pardir,
            )
        )

        self.dir_path = os.path.abspath(
            os.path.join("data_samples", "machine_translation")
        )

        self.pipeline: Pipeline = Pipeline()

    def test_initialize_pipeline(self):
        self.pipeline.set_reader(OntonotesReader())
        self.pipeline.add(
            MachineTranslationProcessor(),
            config={"pretrained_model": "t5-small"},
        )
        self.pipeline.run(self.dir_path)
