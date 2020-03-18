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

from typing import List, Optional, Tuple
from texar.torch import HParams

from forte.evaluation.base import Evaluator
from forte.common import Resources
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Query

from examples.passage_reranker.eval_script import compute_metrics_from_files


class MSMarcoEvaluator(Evaluator[MultiPack]):

    def __init__(self):
        super().__init__()
        self.predicted_results: List[Tuple[str, str, str]] = []
        self._score: Optional[float] = None

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, resources: Resources, configs: HParams):
        self.resource = resources
        self.config = configs

    def consume_next(self, pred_pack, _):
        query_pack = pred_pack.get_pack(self.config.pack_name)
        query = list(query_pack.get_entries_by_type(Query))[0]
        rank = 1
        for pid, _ in query.results.items():
            self.predicted_results.append(
                (query_pack.meta.doc_id, pid, str(rank)))
            rank += 1

    def get_result(self):
        if self._score is None:
            with open(self.config.output_file, "w") as f:
                for result in self.predicted_results:
                    f.write('\t'.join(result) + '\n')

            self._score = compute_metrics_from_files(
                self.config.ground_truth_file, self.config.output_file)
        return self._score
