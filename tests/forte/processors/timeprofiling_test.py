# Copyright 2021 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for time profiling of pipeline.
"""

import unittest
import time

from typing import Any, Dict, Iterator, Optional, Type, Set, List
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.processors.base import PackProcessor, FixedSizeBatchPackingProcessor
from forte.processors.base.batch_processor import Predictor
from ft.onto.base_ontology import Token, Sentence, EntityMention, RelationLink


class DummyPackProcessor(PackProcessor):
    def __init__(self, sleep_time: float):
        super().__init__()
        self.sleep_time: float = sleep_time

    def _process(self, input_pack: DataPack):
        time.sleep(self.sleep_time)


class TestTimeProfiling(unittest.TestCase):
    def setUp(self):
        self.pl = Pipeline[DataPack]()
        self.pl.set_reader(StringReader())
        self.pl.add(DummyPackProcessor(0.9))
        self.pl.add(DummyPackProcessor(0.5))
        self.pl.add(DummyPackProcessor(1.2))

        self.pl.set_profiling()
        self.pl.initialize()

    def test_ner(self):
        sentences = [
            "This tool is called New York .",
            "The goal of this project to help you build NLP " "pipelines.",
            "NLP has never been made this easy before.",
        ]
        document = " ".join(sentences)

        with self.assertLogs("forte.pipeline", level="INFO") as log:
            pack = self.pl.process_one(document)
            self.pl.finish()

        self.assertEqual(len(log.output), 1)
        lines: List[str] = log.output[0].split("\n")
        self.assertEqual(len(lines), 5)
        self.assertEqual("INFO:forte.pipeline:Pipeline Time Profile", lines[0])
        self.assertEqual(
            f"- Reader: {self.pl.reader.component_name}, "
            + f"{self.pl.reader.time_profile} s",
            lines[1],
        )
        for i in range(3):
            self.assertIn(
                f"- Component [{i}]: "
                + f"{self.pl.components[i].name}, "
                + f"{self.pl._profiler[i]} s",
                lines[i + 2],
            )

        self.assertGreater(self.pl.reader.time_profile, 0.0)
        self.assertTrue(all(t > 0.0 for t in self.pl._profiler))


if __name__ == "__main__":
    unittest.main()
