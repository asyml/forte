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
Unit tests for RACEMultiChoiceQAReader.
"""
import json
import os
import unittest
from typing import Iterator, Iterable

from forte.data.data_pack import DataPack
from forte.data.readers import RACEMultiChoiceQAReader
from forte.pipeline import Pipeline
from ft.onto.race_multi_choice_qa_ontology import RaceDocument, Question


class RACEMultiChoiceQAReaderTest(unittest.TestCase):

    def setUp(self):
        self.dataset_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            *([os.path.pardir] * 4),
            'data_samples/race_multi_choice_qa'))

    def test_reader_no_replace_test(self):
        # Read with no replacements
        pipeline = Pipeline()
        reader = RACEMultiChoiceQAReader()
        pipeline.set_reader(reader)
        pipeline.initialize()

        data_packs: Iterable[DataPack] = pipeline.process_dataset(
            self.dataset_path)
        file_paths: Iterator[str] = reader._collect(self.dataset_path)

        count_packs = 0
        for pack, file_path in zip(data_packs, file_paths):
            count_packs += 1
            expected_text: str = ""
            with open(file_path, "r", encoding="utf8", errors='ignore') as file:
                expected = json.load(file)

            articles = list(pack.get(RaceDocument))
            self.assertEqual(len(articles), 1)
            expected_article = expected['article']
            self.assertEqual(articles[0].text, expected_article)
            expected_text += expected_article

            for qid, question in enumerate(pack.get(Question)):
                expected_question = expected['questions'][qid]
                self.assertEqual(question.text, expected_question)
                expected_answers = expected['answers'][qid]
                if not isinstance(expected_answers, list):
                    expected_answers = [expected_answers]
                expected_answers = [reader._convert_to_int(ans)
                                    for ans in expected_answers]
                self.assertEqual(question.answers, expected_answers)
                expected_text += '\n' + expected_question

                for oid, option in enumerate(question.options):
                    expected_option = expected['options'][qid][oid]
                    self.assertEqual(option.text, expected_option)
                    expected_text += '\n' + expected_option

            self.assertEqual(pack.text, expected_text)
        self.assertEqual(count_packs, 2)


if __name__ == "__main__":
    unittest.main()
