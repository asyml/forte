from forte.pipeline import Pipeline

import os
from forte import Pipeline
from forte.data import DataPack
from forte.processors.base import PackProcessor
from forte.data.readers import PlainTextReader
import unittest

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
