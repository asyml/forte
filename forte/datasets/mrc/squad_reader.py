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
from ft.onto.base_ontology import Document, MRCQuestion, Phrase
from ftx.onto.race_qa import Passage

__all__ = [
    "SquadReader",
]


class SquadReader(PackReader):
    r"""Reader for processing Stanford Question Answering Dataset (SQuAD).

    Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset,
    consisting of questions posed by crowdworkers on a set of Wikipedia articles,
    where the answer to every question is a segment of text, or span.

    Dataset can be downloaded at https://rajpurkar.github.io/SQuAD-explorer/.

    SquadReader reads each paragraph in the dataset as a separate Document, and the questions
    are concatenated behind the paragraph, form a Passage.
    Phrase are MRC answers marked as text spans. Each MRCQuestion has a list of answers.
    """

    def _collect(self, file_path: str) -> Iterator[Any]:
        r"""Given file_path to the dataset, return an iterator to every data point in it.

        Args:
            file_path: path to the JSON file

        Returns: QA pairs and the context of a paragraph of a passage in SQuAD dataset.
        """
        with open(file_path, "r", encoding="utf8", errors="ignore") as file:
            jsonf = json.load(file)
            for dic in jsonf["data"]:
                title = dic["title"]
                cnt = 0
                for qa_dic in dic["paragraphs"]:
                    yield title + str(cnt), qa_dic["qas"], qa_dic["context"]
                    cnt += 1

    def _cache_key_function(self, text_file: str) -> str:
        return os.path.basename(text_file)

    def _parse_pack(self, qa_dict: Tuple[str, list, str]) -> Iterator[DataPack]:
        title, qas, context = qa_dict
        context_end = len(context)
        offset = context_end + 1
        text = context

        pack = DataPack()  # one datapack for a context
        for qa in qas:
            if qa["is_impossible"] is True:
                continue
            ques_text = qa["question"]
            ans = qa["answers"]
            text += "\n" + ques_text
            ques_end = offset + len(ques_text)
            question = MRCQuestion(pack, offset, ques_end)
            question.qid = qa["id"]
            offset = ques_end + 1
            for a in ans:
                ans_text = a["text"]
                ans_start = a["answer_start"]
                answer = Phrase(pack, ans_start, ans_start + len(ans_text))
                question.answers.append(answer)

        pack.set_text(text)

        passage = Passage(pack, 0, context_end)
        Document(pack, 0, len(pack.text))

        passage.passage_id = title
        pack.pack_name = title
        yield pack

    @classmethod
    def default_configs(cls):
        return {"file_ext": ".txt"}

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `PlainTextReader` which is
        `ft.onto.base_ontology.Document` with an empty set
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta["ft.onto.base_ontology.Document"] = set()
