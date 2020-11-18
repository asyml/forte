#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from pathlib import Path

from forte.data.data_pack import DataPack
from forte.evaluation.base import Evaluator
from forte.processors.ner_predictor import CoNLLNERPredictor
from ft.onto.base_ontology import Sentence, Token, EntityMention


# TODO: generalize this class to be Forte library code

def _write_tokens_to_file(pred_pack, pred_request,
                          refer_pack, refer_request,
                          output_filename):
    opened_file = open(output_filename, "w+")
    for pred_sentence, tgt_sentence in zip(
            pred_pack.get_data(**pred_request),
            refer_pack.get_data(**refer_request)
    ):

        pred_entity_mention, tgt_entity_mention = \
            pred_sentence["EntityMention"], tgt_sentence["EntityMention"]
        tgt_tokens = tgt_sentence["Token"]

        tgt_ptr, pred_ptr = 0, 0

        for i in range(len(tgt_tokens["text"])):
            w = tgt_tokens["text"][i]
            p = tgt_tokens["pos"][i]
            ch = tgt_tokens["chunk"][i]
            # TODO: This is not correct and probably we need a utility to do
            #       BIO encoding to get ner_type?
            if tgt_ptr < len(tgt_entity_mention["span"]) and \
                    (tgt_entity_mention["span"][tgt_ptr] ==
                     tgt_tokens["span"][i]).all():
                tgt = tgt_entity_mention["ner_type"][tgt_ptr]
                tgt_ptr += 1
            else:
                tgt = "O"

            if pred_ptr < len(pred_entity_mention["span"]) and \
                    (pred_entity_mention["span"][pred_ptr] ==
                     tgt_tokens["span"][i]).all():
                pred = pred_entity_mention["ner_type"][pred_ptr]
                pred_ptr += 1
            else:
                pred = "O"

            opened_file.write(
                "%d %s %s %s %s %s\n" % (i + 1, w, p, ch, tgt, pred)
            )

        opened_file.write("\n")
    opened_file.close()


class CoNLLNEREvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.test_component = CoNLLNERPredictor().name
        self.output_file = "tmp_eval.txt"
        self.score_file = "tmp_eval.score"
        self.scores = {}

    def consume_next(self, pred_pack: DataPack, refer_pack: DataPack):
        pred_getdata_args = {
            "context_type": Sentence,
            "request": {
                EntityMention: {
                    "fields": ["ner_type"],
                },
                Sentence: [],  # span by default
            },
        }

        refer_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {
                    "fields": ["chunk", "pos", "ner"]
                },
                EntityMention: {
                    "fields": ["ner_type"],
                },
                Sentence: [],  # span by default
            }
        }

        _write_tokens_to_file(pred_pack=pred_pack,
                              pred_request=pred_getdata_args,
                              refer_pack=refer_pack,
                              refer_request=refer_getdata_args,
                              output_filename=self.output_file)
        eval_script = \
            Path(os.path.abspath(__file__)).parents[2] / \
            "forte/utils/eval_scripts/conll03eval.v2"
        os.system(f"perl {eval_script} < {self.output_file} > "
                  f"{self.score_file}")
        with open(self.score_file, "r") as fin:
            fin.readline()
            line = fin.readline()
            fields = line.split(";")
            acc = float(fields[0].split(":")[1].strip()[:-1])
            precision = float(fields[1].split(":")[1].strip()[:-1])
            recall = float(fields[2].split(":")[1].strip()[:-1])
            f1 = float(fields[3].split(":")[1].strip())

        self.scores = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def get_result(self):
        return self.scores
