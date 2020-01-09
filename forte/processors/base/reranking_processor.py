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
# pylint: disable=attribute-defined-outside-init
from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data import MultiPack, DataPack
from forte.data.ontology import Query
from forte.processors.base import MultiPackProcessor

__all__ = [
    "RerankingProcessor"
]


class RerankingProcessor(MultiPackProcessor):

    def initialize(self, resources: Resources, configs: HParams):
        self.resources = resources
        self.config = HParams(configs, self.default_hparams())

    def get_matching_score(self, query_pack: DataPack, document_pack: DataPack):
        """
        Scoring function between a query and a document
        Args:
            query_pack: pack associated with the query
            document_pack: pack associated with the document
        Returns:
            A positive score between query text and the document text that is
            used for ranking documents for each query
        """
        raise NotImplementedError

    def _process(self, input_pack: MultiPack):
        query_pack_name = self.config.query_pack_name

        query_pack = input_pack.get_pack(self.config.query_pack_name)
        query_entry = list(query_pack.get_entries(Query))[0]

        packs = {}
        doc_scores = []
        for doc_id in input_pack.pack_names:
            if doc_id == query_pack_name:
                continue

            document_pack = input_pack.get_pack(doc_id)

            match_score = self.get_matching_score(query_pack, document_pack)

            query_entry.update_passage({doc_id: match_score})
            doc_scores.append((match_score, doc_id))
            packs[doc_id] = document_pack
