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
Unit tests for SquadReader.
"""
import json
import os
import unittest
from typing import Iterable

from forte.data.data_pack import DataPack
from forte.datasets.mrc.squad_reader import SquadReader
from forte.pipeline import Pipeline
from ft.onto.base_ontology import MRCQuestion
from ftx.onto.race_qa import Passage


class SquadReaderTest(unittest.TestCase):
    def setUp(self):
        self.dataset_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                *([os.path.pardir] * 4),
                "data_samples/squad_v2.0/dev-v2.0-sample.json"
            )
        )

    def test_reader_no_replace_test(self):
        # Read with no replacements
        pipeline = Pipeline()
        reader = SquadReader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        data_packs: Iterable[DataPack] = pipeline.process_dataset(
            self.dataset_path
        )
        file_path: str = self.dataset_path
        expected_file_dict = {}
        with open(file_path, "r", encoding="utf8", errors="ignore") as file:
            expected_json = json.load(file)
            for dic in expected_json["data"]:
                title = dic["title"]
                cnt = 0
                for qa_dic in dic["paragraphs"]:
                    expected_file_dict[
                        title + str(cnt)
                    ] = qa_dic  # qas, context
                    cnt += 1

        count_packs = 0
        for pack in data_packs:
            count_packs += 1
            expected_text: str = ""
            expected = expected_file_dict[pack.pack_name]

            passage = list(pack.get(Passage))
            self.assertEqual(len(passage), 1)
            expected_context = expected["context"]
            self.assertEqual(passage[0].text, expected_context)
            expected_text += expected_context

            for qid, question in enumerate(pack.get(MRCQuestion)):
                expected_qa = expected["qas"][qid]
                expected_question = expected_qa["question"]
                expected_answers = expected_qa["answers"]
                self.assertEqual(question.text, expected_question)
                if not isinstance(expected_answers, list):
                    expected_answers = [expected_answers]
                answers = question.answers

                for answer, expected_answer in zip(answers, expected_answers):
                    self.assertEqual(answer.text, expected_answer["text"])
                expected_text += "\n" + expected_question

            self.assertEqual(pack.text, expected_text)


if __name__ == "__main__":
    unittest.main()
