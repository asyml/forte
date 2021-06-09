# Copyright 2019 The Forte Authors. All Rights Reserved.
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
Unit tests for OpenIEReader.
"""
import os
import unittest
from typing import Iterator, Iterable, List

from forte.data.data_pack import DataPack
from forte.data.readers import OpenIEReader
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, RelationLink


class OpenIEReaderTest(unittest.TestCase):
    def setUp(self):
        # Define and config the pipeline.
        self.dataset_path: str = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                *([os.path.pardir] * 4),
                "data_samples/openie"
            )
        )

        self.pipeline: Pipeline = Pipeline[DataPack]()
        self.reader: OpenIEReader = OpenIEReader()
        self.pipeline.set_reader(self.reader)
        self.pipeline.initialize()

    def test_process_next(self):
        data_packs: Iterable[DataPack] = self.pipeline.process_dataset(
            self.dataset_path
        )
        file_paths: Iterator[str] = self.reader._collect(self.dataset_path)

        count_packs: int = 0

        for pack, file_path in zip(data_packs, file_paths):
            count_packs += 1
            with open(file_path, "r", encoding="utf8", errors="ignore") as file:
                expected_doc = file.read()

            lines: List[str] = expected_doc.split("\n")
            actual_sentences: Iterator[Sentence] = pack.get(Sentence)

            for line, actual_sentence in zip(lines, actual_sentences):
                segments: List[str] = line.strip().split("\t")

                # Test sentence.
                expected_sentence: str = segments[0]
                self.assertEqual(actual_sentence.text, expected_sentence)

                # Test head predicate.
                link: RelationLink = list(
                    pack.get(RelationLink, actual_sentence)
                )[0]

                self.assertEqual(link.rel_type, segments[2])
                self.assertEqual(link.get_parent().text, segments[3])
                self.assertEqual(link.get_child().text, segments[4])

        self.assertEqual(count_packs, 1)


if __name__ == "__main__":
    unittest.main()
