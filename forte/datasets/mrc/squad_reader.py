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
import os
import json
from typing import Any, Iterator, Dict, Set, Tuple

from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from ft.onto.base_ontology import Document, MRCQuestion, MRCAnswer
from ftx.onto.race_qa import Passage

__all__ = [
    "SquadReader",
]


class SquadReader(PackReader):
    r"""Reader for processing Stanford Question Answering Dataset (SQuAD).

    """

    def _collect(self, file_path) -> Iterator[Any]:  # type: ignore
        r"""Should be called with param ``text_directory`` which is a path to a
        folder containing txt files.

        Args:
            text_directory: text directory containing the files.

        Returns: Iterator over paths to .txt files
        """
        with open(file_path, "r", encoding="utf8", errors="ignore") as file:
            jsonf = json.load(file)
            for dic in jsonf["data"]:
                title = dic["title"]
                cnt = 0
                for qa_dic in dic["paragraphs"]:
                    yield title+str(cnt), qa_dic["qas"], qa_dic["context"]
                    cnt += 1

    def _cache_key_function(self, text_file: str) -> str:
        return os.path.basename(text_file)

    # pylint: disable=unused-argument
    def text_replace_operation(self, text: str):
        return []

    def _parse_pack(self, qa_dict: Tuple[str, list, str]) -> Iterator[DataPack]:
        title, qas, context = qa_dict
        context_end = len(context)
        offset = context_end+1
        text = context

        pack = DataPack() # one datapack for a context
        for qa in qas:
            if qa["is_impossible"] == True:
                continue
            ques_text = qa["question"]
            ans = qa["answers"]
            text += "\n" + ques_text
            ques_end = offset + len(ques_text)
            question = MRCQuestion(pack, offset, ques_end)
            offset = ques_end+1
            for a in ans:
                ans_text = a["text"]
                ans_start = a["answer_start"]
                answer = MRCAnswer(pack, ans_start, ans_start+len(ans_text))
                question.answers.append(answer)

        pack.set_text(text, replace_func=self.text_replace_operation)

        Document(pack, 0, context_end)
        passage = Passage(pack, 0, len(pack.text))

        passage.passage_id = title
        pack.pack_name = title
        yield pack

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config["file_ext"] = ".txt"
        return config

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `PlainTextReader` which is
        `ft.onto.base_ontology.Document` with an empty set
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Document"] = set()

