# Copyright 2021 The Forte Authors. All Rights Reserved.
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
"""Evaluator for Conll03 NER tag."""
import os
from typing import Dict
from pathlib import Path

from ft.onto.base_ontology import Sentence
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.evaluation.base import Evaluator
from forte.utils import get_class
from forte.datasets.conll.conll_utils import write_tokens_to_file


class CoNLLNEREvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.output_file = "tmp_eval.txt"
        self.score_file = "tmp_eval.score"
        self.scores: Dict[str, float] = {}
        if os.path.isfile(self.output_file):
            os.remove(self.output_file)

    def initialize(self, resources: Resources, configs: Config):
        # pylint: disable=attribute-defined-outside-init,unused-argument
        r"""Initialize the evaluator with `resources` and `configs`.
        This method is called by the pipeline during the initialization.

        Args:
            resources (Resources): An object of class
                :class:`~forte.common.Resources` that holds references to
                objects that can be shared throughout the pipeline.
            configs (Config): A configuration to initialize the
                evaluator. This evaluator is expected to hold the
                following (key, value) pairs
                - `"entry_type"` (str): The entry to be evaluated.
                - `"tagging_unit"` (str): The tagging unit that the evaluation
                is performed on. e.g. `"ft.onto.base_ontology.Sentence"`
                - `"attribute"` (str): The attribute of the entry to be
                evaluated.

        """
        super().initialize(resources, configs)
        self.entry_type = get_class(configs.entry_type)
        self.tagging_unit = get_class(configs.tagging_unit)
        self.attribute = configs.attribute
        self.__eval_script = configs.eval_script

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update(
            {
                "entry_type": None,
                "tagging_unit": None,
                "attribute": "",
                "eval_script": str(
                    Path(os.path.abspath(__file__)).parents[1]
                    / "utils/eval_scripts/conll03eval.v2"
                ),
            }
        )
        return config

    def consume_next(self, pred_pack: DataPack, ref_pack: DataPack):
        refer_getdata_args = {
            "context_type": Sentence,
            "request": {
                self.tagging_unit: {"fields": ["ner"]},
                self.entry_type: {
                    "fields": [self.attribute],
                },
            },
        }
        write_tokens_to_file(
            pred_pack,
            ref_pack,
            refer_getdata_args,
            self.tagging_unit,
            self.entry_type,
            self.attribute,
            self.output_file,
        )

    def get_result(self) -> Dict:
        eval_call = (
            f"perl {self.__eval_script} < {self.output_file} > "
            f"{self.score_file}"
        )

        call_return = os.system(eval_call)
        if call_return == 0:
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
            return self.scores
        else:
            raise RuntimeError(
                f"Error running eval script, return code is {call_return} "
                f"when running the command {eval_call}"
            )
