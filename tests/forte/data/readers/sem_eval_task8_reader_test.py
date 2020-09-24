# Copyright 2020 The Forte Authors. All Rights Reserved.
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
Unit tests for SemEvalTask8Reader.
"""
import os
import unittest
from typing import Iterator, Iterable

from forte.data.data_pack import DataPack
from forte.data.readers import SemEvalTask8Reader
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, RelationLink


class SemEvalTask8ReaderTest(unittest.TestCase):
    def setUp(self):
        self.dataset_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            *([os.path.pardir] * 4),
            'data_samples/sem_eval_task8'))

    def test_reader_no_replace_test(self):
        pipeline = Pipeline[DataPack]()
        reader = SemEvalTask8Reader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        expected_sents = [
            "The system as described above has its greatest application" +
            " in an arrayed configuration of antenna elements.",
            "The child was carefully wrapped and bound into the cradle by means of a cord.",
            "The author of a keygen uses a disassembler to look at the raw assembly code.",
            "A misty ridge uprises from the surge.",
            "The student association is the voice of the undergraduate student" +
            " population of the State University of New York at Buffalo."
        ]
        expected_relations = [
            ("Component-Whole", "elements", "configuration"),
            ("Other", "child", "cradle"),
            ("Instrument-Agency", "disassembler", "author"),
            ("Other", "ridge", "surge"),
            ("Member-Collection", "student", "association")
        ]
        expected_text = " ".join(expected_sents) + " "

        data_packs: Iterable[DataPack] = pipeline.process_dataset(
            self.dataset_path)

        count_packs = 0
        for pack in data_packs:
            count_packs += 1

            sents = list(pack.get(Sentence))
            relations = list(pack.get(RelationLink))

            for s, r in zip(sents, relations):
                self.assertIn(s.text, expected_sents)
                index = expected_sents.index(s.text)
                r = pack.get(RelationLink, s)
                r = next(r)
                self.assertEqual(r.rel_type,
                                expected_relations[index][0])
                self.assertEqual(r.get_parent().text,
                                expected_relations[index][1])
                self.assertEqual(r.get_child().text,
                                expected_relations[index][2])

            self.assertEqual(expected_text, pack.text)

        self.assertEqual(count_packs, 1)


if __name__ == "__main__":
    unittest.main()
