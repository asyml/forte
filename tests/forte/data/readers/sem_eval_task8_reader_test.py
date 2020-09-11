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
Unit tests for SemEvalTask8Reader.
"""
import json
import os
import logging
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

        data_packs: Iterable[DataPack] = pipeline.process_dataset(
            self.dataset_path)
        file_paths: Iterator[str] = reader._collect(self.dataset_path)

        count_packs = 0

        for pack, file_path in zip(data_packs, file_paths):
            count_packs += 1

            expected_text: str = ""
            expected_sents = []
            expected_relations = []

            fp = open(file_path, 'r', encoding='utf-8')
            while True:
                sent_line = fp.readline()
                if not sent_line:
                    break
                if len(sent_line.split()) == 0:
                    continue
                relation_line = fp.readline()
                # command line is not used
                _ = fp.readline()

                sent_line = sent_line[sent_line.find('"') + 1:
                                    sent_line.rfind('"')]
                e1 = sent_line[sent_line.find("<e1>"):
                                    sent_line.find("</e1>") + 5]
                e2 = sent_line[sent_line.find("<e2>"):
                                    sent_line.find("</e2>") + 5]
                sent_line = sent_line.replace(e1, e1[4:-5])
                sent_line = sent_line.replace(e2, e2[4:-5])
                e1 = e1[4:-5]
                e2 = e2[4:-5]
                expected_text += sent_line + " "

                pair = relation_line[relation_line.find("(") + 1:
                            relation_line.find(")")]
                if "," in pair:
                    if pair.split(",")[0] == 'e1':
                        parent = e1
                        child = e2
                    else:
                        parent = e2
                        child = e1
                    rel_type = relation_line[:relation_line.find("(")]
                else:
                    parent, child = e1, e2
                    rel_type = relation_line.strip()

                expected_sents.append(sent_line)
                expected_relations.append((parent, child, rel_type))

            sents = list(pack.get(Sentence))
            relations = list(pack.get(RelationLink))

            for s, r in zip(sents, relations):
                self.assertIn(s.text, expected_sents)
                index = expected_sents.index(s.text)
                r = pack.get(RelationLink, s)
                r = next(r)
                self.assertEqual(r.get_parent().text,
                                expected_relations[index][0])
                self.assertEqual(r.get_child().text,
                                expected_relations[index][1])           
                self.assertEqual(r.rel_type,
                                expected_relations[index][2])

            self.assertEqual(expected_text, pack.text)

        self.assertEqual(count_packs, 1)


if __name__ == "__main__":
    unittest.main()