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

from typing import List, Optional, Tuple

from forte.common import ProcessExecutionException
from forte.data.data_pack import DataPack
from forte.evaluation.base import Evaluator
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Query

from examples.passage_ranker.eval_script import compute_metrics_from_files


class MSMarcoEvaluator(Evaluator[MultiPack]):
    def __init__(self):
        super().__init__()
        self.predicted_results: List[Tuple[str, str, str]] = []
        self._score: Optional[float] = None

    def consume_next(self, pred_pack: MultiPack, _):
        query_pack: DataPack = pred_pack.get_pack(self.configs.pack_name)
        query = list(query_pack.get(Query))[0]
        rank = 1
        for pid, _ in query.results.items():
            doc_id: Optional[str] = query_pack.meta.doc_id
            if doc_id is None:
                raise ProcessExecutionException(
                    'Doc ID of the query pack is not set, '
                    'please double check the reader.')
            self.predicted_results.append((doc_id, pid, str(rank)))
            rank += 1

    def get_result(self):
        curr_dir = os.path.dirname(__file__)
        output_file = os.path.join(curr_dir, self.configs.output_file)
        gt_file = os.path.join(curr_dir, self.configs.ground_truth_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if self._score is None:
            with open(output_file, "w") as f:
                for result in self.predicted_results:
                    f.write('\t'.join(result) + '\n')

            self._score = compute_metrics_from_files(gt_file, output_file)
        return self._score

    @classmethod
    def default_configs(cls):
        return {
            'pack_name': None,
            'output_file': None,
            'ground_truth_file': None,
        }
