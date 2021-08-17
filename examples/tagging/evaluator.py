#  Copyright 2020 The Forte Authors. All Rights Reserved.
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
# pylint: disable-msg=too-many-locals
"""Evaluator for Conll03 NER tag."""
import os
from pathlib import Path
from forte.data.base_pack import PackType
from forte.evaluation.base import Evaluator
from forte.data.extractor.utils import bio_tagging
from ft.onto.base_ontology import Sentence, Token, EntityMention


def _post_edit(element):
    if element[0] is None:
        return "O"
    return "%s-%s" % (element[1], element[0].ner_type)


def _get_tag(data, pack):
    based_on = [pack.get_entry(x) for x in data["Token"]["tid"]]
    entry = [pack.get_entry(x) for x in data["EntityMention"]["tid"]]
    tag = bio_tagging(based_on, entry)
    tag = [_post_edit(x) for x in tag]
    return tag


def _write_tokens_to_file(
    pred_pack, pred_request, refer_pack, refer_request, output_filename
):
    opened_file = open(output_filename, "w+")
    for pred_data, refer_data in zip(
        pred_pack.get_data(**pred_request), refer_pack.get_data(**refer_request)
    ):
        pred_tag = _get_tag(pred_data, pred_pack)
        refer_tag = _get_tag(refer_data, refer_pack)
        words = refer_data["Token"]["text"]
        pos = refer_data["Token"]["pos"]
        chunk = refer_data["Token"]["chunk"]

        for i, (word, position, chun, tgt, pred) in enumerate(
            zip(words, pos, chunk, refer_tag, pred_tag), 1
        ):
            opened_file.write(
                "%d %s %s %s %s %s\n" % (i, word, position, chun, tgt, pred)
            )
        opened_file.write("\n")
    opened_file.close()


class CoNLLNEREvaluator(Evaluator):
    """Evaluator for Conll NER task."""

    def __init__(self):
        super().__init__()
        # self.test_component = CoNLLNERPredictor().name
        self.output_file = "tmp_eval.txt"
        self.score_file = "tmp_eval.score"
        self.scores = {}

    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        pred_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {"fields": ["chunk", "pos"]},
                EntityMention: {
                    "fields": ["ner_type"],
                },
                Sentence: [],  # span by default
            },
        }

        refer_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {"fields": ["chunk", "pos", "ner"]},
                EntityMention: {
                    "fields": ["ner_type"],
                },
                Sentence: [],  # span by default
            },
        }

        _write_tokens_to_file(
            pred_pack=pred_pack,
            pred_request=pred_getdata_args,
            refer_pack=ref_pack,
            refer_request=refer_getdata_args,
            output_filename=self.output_file,
        )
        eval_script = (
            Path(os.path.abspath(__file__)).parents[2]
            / "forte/utils/eval_scripts/conll03eval.v2"
        )
        os.system(
            f"perl {eval_script} < {self.output_file} > " f"{self.score_file}"
        )
        with open(self.score_file, "r") as fin:
            fin.readline()
            line = fin.readline()
            fields = line.split(";")
            acc = float(fields[0].split(":")[1].strip()[:-1])
            precision = float(fields[1].split(":")[1].strip()[:-1])
            recall = float(fields[2].split(":")[1].strip()[:-1])
            f_1 = float(fields[3].split(":")[1].strip())

        self.scores = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f_1,
        }

    def get_result(self):
        return self.scores
