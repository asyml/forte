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
The reader that reads RACE multi choice QA data into Datapacks.
"""
import os
import json

from typing import Any, Iterator, List

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.readers.base_reader import PackReader
from ft.onto.race_multi_choice_qa_ontology import (
    RaceDocument, Passage, Question, Option)

__all__ = [
    "RACEMultiChoiceQAReader",
]


class RACEMultiChoiceQAReader(PackReader):
    r""":class:`RACEMultiChoiceQAReader` is designed to read in RACE multi
    choice qa dataset.
    """

    def _collect(self, json_directory) -> Iterator[Any]:  # type: ignore
        r"""Should be called with param ``json_directory`` which is a path to a
        folder containing json files.

        Args:
            json_directory: directory containing the json files.

        Returns: Iterator over paths to .json files
        """
        return dataset_path_iterator(json_directory, "")

    def _cache_key_function(self, json_file: str) -> str:
        return os.path.basename(json_file)

    def _convert_to_int(self, ch: Any) -> int:
        if isinstance(ch, int):
            return ch
        if isinstance(ch, str):
            return ord(ch.lower()) - 97
        raise ValueError("Wrong datatype for Answers: expected int or str, "
                         f"got {type(ch).__name__}")

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        with open(file_path, "r", encoding="utf8", errors='ignore') as file:
            dataset = json.load(file)

            pack = DataPack()
            text: str = dataset['article']
            article_end = len(text)
            offset = article_end + 1

            for qid, ques_text in enumerate(dataset['questions']):
                text += '\n' + ques_text
                ques_end = offset + len(ques_text)
                question = Question(pack, offset, ques_end)
                offset = ques_end + 1

                options: List[Option] = []
                options_text = dataset['options'][qid]
                for option_text in options_text:
                    text += '\n' + option_text
                    option_end = offset + len(option_text)
                    option = Option(pack, offset, option_end)
                    options.append(option)
                    offset = option_end + 1
                question.options = options

                answers = dataset['answers'][qid]
                if not isinstance(answers, list):
                    answers = [answers]
                answers = [self._convert_to_int(ans) for ans in answers]
                question.answers = answers

            pack.set_text(text, replace_func=self.text_replace_operation)

            RaceDocument(pack, 0, article_end)

            passage_id: str = dataset['id']
            passage = Passage(pack, 0, len(pack.text))
            passage.passage_id = passage_id

            pack.meta.doc_id = passage_id
            yield pack
