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
from typing import Dict, Any

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Query
from forte.indexers.elastic_indexer import ElasticSearchIndexer
from forte.processors.base import MultiPackProcessor
from ft.onto.base_ontology import Document

__all__ = [
    "ElasticSearchProcessor"
]


class ElasticSearchProcessor(MultiPackProcessor):
    r"""This processor searches for relevant documents for a query"""

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, resources: Resources, configs: Config):
        self.resources = resources
        self.config = configs
        self.index = ElasticSearchIndexer(config=self.config.index_config)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "query_pack_name": "query",
            "index_config": ElasticSearchIndexer.default_configs(),
            "field": "content",
            "response_pack_name_prefix": "passage"
        })
        return config

    def _process(self, input_pack: MultiPack):
        r"""Searches ElasticSearch indexer to fetch documents for a query. This
        query should be contained in the input multipack with name
        `self.config.query_pack_name`.

        This method adds new packs to `input_pack` containing the retrieved
        results. Each result is added as a `ft.onto.base_ontology.Document`.

        Args:
             input_pack: A multipack containing query as a pack.
        """
        query_pack = input_pack.get_pack(self.config.query_pack_name)

        # ElasticSearchQueryCreator adds a Query entry to query pack. We now
        # fetch it as the first element.
        first_query: Query = query_pack.get_single(Query)  # type: ignore
        # pylint: disable=isinstance-second-argument-not-valid-type
        # TODO: until fix: https://github.com/PyCQA/pylint/issues/3507
        if not isinstance(first_query, Dict):
            raise ValueError(
                "The query to the elastic indexer need to be a dictionary.")
        results = self.index.search(first_query.value)
        hits = results["hits"]["hits"]

        for idx, hit in enumerate(hits):
            document = hit["_source"]
            first_query.add_result(document["doc_id"], hit["_score"])

            pack: DataPack = input_pack.add_pack(
                f"{self.config.response_pack_name_prefix}_{idx}"
            )
            pack.pack_name = document["doc_id"]

            content = document[self.config.field]
            pack.set_text(content)

            Document(pack=pack, begin=0, end=len(content))
