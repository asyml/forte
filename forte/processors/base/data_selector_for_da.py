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
# pylint: disable=attribute-defined-outside-init
# pylint: disable=useless-super-delegation
"""
An elastic search indexer needs to be created first
in order to perform data selection.
Refer to examples/data_augmentation/data_select_index_pipeline.py
for indexer creation.
"""
from abc import ABC
from typing import Iterator, Any, Dict, Optional

from forte.common.resources import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from forte.indexers.elastic_indexer import ElasticSearchIndexer

__all__ = [
    "BaseElasticSearchDataSelector",
    "RandomDataSelector",
    "QueryDataSelector",
]


class BaseDataSelector(PackReader, ABC):
    r"""A base data selector for data augmentation.
    It is a reader that selects a subset from the dataset and yields datapacks.
    """


class BaseElasticSearchDataSelector(BaseDataSelector):
    r"""The base elastic search indexer for data selector. This class creates
    an :class:`~forte.indexers.elastic_indexer.ElasticSearchIndexer`
    and searches for documents according to the user-provided search keys.
    Currently supported search criteria: random-based and query-based. It
    then yields the corresponding datapacks of the selected documents.
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.index = ElasticSearchIndexer(config=self.configs.index_config)

    def _create_search_key(self, data: Optional[str]) -> Dict[str, Any]:
        raise NotImplementedError

    def _collect(self, *args, **kwargs) -> Iterator[str]:
        raise NotImplementedError

    def _parse_pack(self, pack_info: str) -> Iterator[DataPack]:
        pack: DataPack = DataPack.deserialize(pack_info)
        yield pack

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "index_config": ElasticSearchIndexer.default_configs(),
        })
        return config


class QueryDataSelector(BaseElasticSearchDataSelector):

    def _collect(self, *args, **kwargs) -> Iterator[str]:
        # pylint: disable = unused-argument
        r"""Iterator over query text files in the data_source.
        Then search for documents using indexer.

        Args:
            args: args[0] is the path to the query file
            kwargs:

        Returns: Selected document's original datapack.
        """
        data_path: str = args[0]
        with open(data_path, 'r') as file:
            for line in file:
                query: Dict = self._create_search_key(line.strip())
                results = self.index.search(query)
                hits = results["hits"]["hits"]

                for _, hit in enumerate(hits):
                    document = hit["_source"]
                    yield document["pack_info"]

    def _create_search_key(self, data: str) -> Dict[str, Any]:  # type: ignore
        r"""Create a search dict for elastic search indexer.
        Args:
             text: str
                A string which will be looked up for in the corpus under field
                name `field`. It can be passed in `config` during initialize.
                If `config` does not contain `field`, we set it to "content".

        Returns: A dict that specifies the query match field.
        """
        return {"query":
                    {"match": {self.configs["field"]: data}},
                "size": self.configs["size"]}

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "size": 1000,
            "field": "content",
        })
        return config


class RandomDataSelector(BaseElasticSearchDataSelector):
    def _collect(self, *args, **kwargs) -> Iterator[str]:
        # pylint: disable = unused-argument
        r"""random select `size` documents from the indexer.
        Returns: Selected document's original datapack.
        """
        query: Dict = self._create_search_key()
        results = self.index.search(query)
        hits = results["hits"]["hits"]

        for _, hit in enumerate(hits):
            document = hit["_source"]
            yield document["pack_info"]

    def _create_search_key(self) -> Dict[str, Any]:  # type: ignore
        return {
            "size": self.configs["size"],
            "query": {
                "function_score": {
                    "functions": [
                        {
                            "random_score": {
                                "seed": "1477072619038",
                                "field": "_seq_no"
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
            "size": 1000000
        })
        return config
