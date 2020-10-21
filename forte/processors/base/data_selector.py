# Copyright 2020 The Forte Authors. All Rights Reserved.
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

"""
A data selector for data augmentation.
It is a reader that search documents from elastic search indexer and yield datapacks.
An elastic search indexer for data selector needs to be created first.
Refer to data_augment/data_select_index_pipeline.py for indexer generation.
"""

from typing import Iterator, Any, Dict, Tuple, Optional

from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import (
    Token, Sentence, Document
)
from forte.common.resources import Resources
from forte.common.configuration import Config
from forte.indexers.elastic_indexer import ElasticSearchIndexer


__all__ = [
    "BaseElasticSearchDataSelector",
    "RandomDataSelector",
    "QueryDataSelector",
]


class BaseElasticSearchDataSelector(PackReader):
    r"""
    The base data selector reader for data augmentation.
    This class creates an ElasticSearchIndexer and search for document
    according to the search key. It then generate Datapacks.
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.index = ElasticSearchIndexer(config=self.config.index_config)

    def _create_search_key(self, data: Optional[str]) -> Dict[str,Any]:
        raise NotImplementedError

    def _collect(self, *args, **kwargs) -> Iterator[str]:
        raise NotImplementedError

    def _parse_pack(self, doc_info: Tuple[str, str]) -> Iterator[DataPack]:
        data_pack: DataPack = self.new_pack()
        doc_id, doc_text = doc_info

        data_pack.pack_name = doc_id
        data_pack.set_text(doc_text)
        Document(data_pack, 0, len(doc_text))

        yield data_pack

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "index_config": ElasticSearchIndexer.default_configs(),
        })
        return config


class QueryDataSelector(BaseElasticSearchDataSelector):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

    def _collect(self, *args, **kwargs) -> Iterator[Tuple[str, str]]:
        # pylint: disable = unused-argument
        r"""Iterator over query text files in the data_source.
        Then search for documents using indexer.

        Args:
            args: args[0] is the path to the query file
            kwargs:

        Returns: Selected document id and document content.
        """
        data_path: str = args[0]
        with open(data_path, 'r') as file:
            for line in file:
                query: Dict = self._create_search_key(line.strip())
                results = self.index.search(query)
                hits = results["hits"]["hits"]

                for idx, hit in enumerate(hits):
                    document = hit["_source"]
                    yield document["doc_id"], document[self.config["field"]]

    def _create_search_key(self, data: str) -> Dict[str,Any]:
        r"""Create a search dict for elastic search indexer.
        Args:
             text: str
                A string which will be looked up for in the corpus under field
                name `field`. `field` can be passed in a `config` during initialize.
                If `config` does not contain `field`, we will set it to "content".

        Returns: A dict that specifies the query match field.
        """
        return {"query":
                    {"match": {self.config["field"]: data}},
                "size": self.config["size"]}

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "size": 1000,
            "field": "content",
        })
        return config


class RandomDataSelector(BaseElasticSearchDataSelector):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

    def _collect(self, *args, **kwargs) -> Iterator[Tuple[str, str]]:
        # pylint: disable = unused-argument
        r"""random select num_of_doc documents from the indexer.
        Returns: Selected document id and document content.
        """
        for _ in range(self.config["num_of_doc"]):
            query: Dict = self._create_search_key()
            results = self.index.search(query)
            hits = results["hits"]["hits"]

            for idx, hit in enumerate(hits):
                document = hit["_source"]
                yield document["doc_id"], document[self.config["field"]]

    def _create_search_key(self) -> Dict[str,Any]:
        return {
           "size": self.config["size"],
           "query": {
              "function_score": {
                 "functions": [
                    {
                       "random_score": {
                          "seed": "1477072619038"
                       }
                    }
                 ]
              }
           }
        }

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "size": 1000000,
        })
        return config
