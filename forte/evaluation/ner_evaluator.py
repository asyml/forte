import os
from pathlib import Path
from typing import Optional, Dict

from texar.torch import HParams

from forte.evaluation.base import Evaluator
from forte.data.data_pack import DataPack
from forte.data.datasets.conll import conll_utils
from forte.processors import CoNLLNERPredictor
from ft.onto.base_ontology import Sentence, Token


class CoNLLNEREvaluator(Evaluator):
    def __init__(self, config: Optional[HParams] = None):
        super().__init__(config)
        self.test_component = CoNLLNERPredictor().component_name
        self.output_file = "tmp_eval.txt"
        self.score_file = "tmp_eval.score"
        self.scores: Dict[str, float] = {}

    def consume_next(self, pred_pack: DataPack, refer_pack: DataPack):
        pred_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {
                    "fields": ["ner"],
                },
                Sentence: [],  # span by default
            },
        }

        refer_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {
                    "fields": ["chunk", "pos", "ner"]},
                Sentence: [],  # span by default
            }
        }

        conll_utils.write_tokens_to_file(pred_pack=pred_pack,
                                         pred_request=pred_getdata_args,
                                         refer_pack=refer_pack,
                                         refer_request=refer_getdata_args,
                                         output_filename=self.output_file)
        eval_script = \
            Path(os.path.abspath(__file__)).parents[1] / \
            "utils/eval_scripts/conll03eval.v2"
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