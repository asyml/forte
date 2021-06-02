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
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Query
from forte.processors.base import MultiPackProcessor
from forte.utils import create_class_with_kwargs
from ft.onto.base_ontology import Document

__all__ = [
    "SearchProcessor"
]


class SearchProcessor(MultiPackProcessor):
    r"""This processor searches for relevant documents for a query"""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        # Replace explicit class with configuration class name.
        self.index = create_class_with_kwargs(
            self.configs.indexer_class,
            class_args={
                'config': self.configs.indexer_configs
            }
        )
        self.index.load(self.configs.model_dir)
        self.k = self.configs.k or 5

    def _process(self, input_pack: MultiPack):
        query_pack = input_pack.get_pack(self.configs.query_pack_name)
        first_query = list(query_pack.get(Query))[0]
        results = self.index.search(first_query.value, self.k)
        documents = [r[1] for result in results for r in result]

        packs = {}
        for i, doc in enumerate(documents):
            pack = input_pack.add_pack()
            pack.set_text(doc)

            Document(pack, 0, len(doc))
            packs[self.configs.response_pack_name_prefix + f'_{i}'] = pack

        input_pack.update_pack(packs)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            'model_dir': None,
            'response_pack_name_prefix': 'doc',
            'indexer_class': 'forte_wrapper.faiss.embedding_based_indexer'
                             '.EmbeddingBasedIndexer',
            'indexer_configs': {
                "index_type": "GpuIndexFlatIP",
                "dim": 768,
                "device": "gpu0"
            }
        })
        return config
