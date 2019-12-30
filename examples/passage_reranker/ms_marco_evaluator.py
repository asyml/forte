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

from texar.torch import HParams

from forte.common import Evaluator, Resources
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Query

from eval_script import compute_metrics_from_files


class MSMarcoEvaluator(Evaluator[MultiPack]):

    def __init__(self):
        super().__init__()
        self.pred_packs = []
        self.ref_packs = []
        self._score = None

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, resource: Resources, configs: HParams):
        self.resource = resource
        self.config = configs

    def consume_next(self, pred_pack, _):
        self.pred_packs.append(pred_pack)

    def get_result(self):
        if self._score is None:
            for pred_pack in self.pred_packs:
                query_pack = pred_pack.get_pack(self.config.pack_name)
                query = list(query_pack.get_entries_by_type(Query))[0]
                with open(self.config.output_file, "a") as f:
                    rank = 1
                    for pid, _ in query.passages.items():
                        f.write('\t'.join(
                            [query_pack.meta.doc_id, pid, str(rank)]) + '\n')
                        rank += 1
            self._score = compute_metrics_from_files(
                self.config.ground_truth_file, self.config.output_file)
        return self._score
