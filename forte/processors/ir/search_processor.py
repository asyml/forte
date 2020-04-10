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
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Query
from forte.processors.base import MultiPackProcessor
from forte.indexers.embedding_based_indexer import EmbeddingBasedIndexer

from ft.onto.base_ontology import Document

__all__ = [
    "SearchProcessor"
]


class SearchProcessor(MultiPackProcessor):
    r"""This processor searches for relevant documents for a query"""

    def __init__(self) -> None:
        super().__init__()

        self.index = EmbeddingBasedIndexer(config={
            "index_type": "GpuIndexFlatIP",
            "dim": 768,
            "device": "gpu0"
        })

    def initialize(self, resources: Resources, configs: Config):
        self.resources = resources
        self.config = configs
        self.index.load(self.config.model_dir)
        self.k = self.config.k or 5

    def _process(self, input_pack: MultiPack):
        query_pack = input_pack.get_pack(self.config.query_pack_name)
        first_query = list(query_pack.get_entries(Query))[0]
        results = self.index.search(first_query.value, self.k)
        documents = [r[1] for result in results for r in result]

        packs = {}
        for i, doc in enumerate(documents):
            pack = DataPack()
            document = Document(pack=pack, begin=0, end=len(doc))
            pack.add_entry(document)
            pack.set_text(doc)
            packs[self.config.response_pack_name[i]] = pack

        input_pack.update_pack(packs)
