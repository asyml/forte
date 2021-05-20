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
from abc import ABC
from typing import Dict, Any, List

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.indexers.elastic_indexer import ElasticSearchIndexer
from forte.processors.base import IndexProcessor

__all__ = [
    "ElasticSearchTextIndexProcessor",
    "ElasticSearchPackIndexProcessor",
]


# pylint: disable=attribute-defined-outside-init


class ElasticSearchIndexerBase(IndexProcessor, ABC):
    r"""This processor implements the basic functions to add the data packs
    into an `Elasticsearch` index."""

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.indexer = ElasticSearchIndexer(self.configs.indexer.hparams)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                "batch_size": 128,
                "fields": "content",
                "indexer": {
                    "name": "ElasticSearchIndexer",
                    "hparams": ElasticSearchIndexer.default_configs(),
                    "kwargs": {
                        "request_timeout": 10,
                        "refresh": False
                    }
                }
            }

        Here:

        `"batch_size"`: int
            Number of examples that will be bulk added to `Elasticsearch` index

        `"fields"`: str, list
            Field name that will be used as a key while indexing the document

        `"indexer"`: dict

            `"name"`: str
                Name of Indexer to be used.

            `"hparams"`: dict
                Hyperparameters to be used for the index. See
                :meth:`ElasticSearchIndexer.default_hparams` for more details

            `"kwargs"`: dict
                Keyword arguments that will be passed to
                :meth:`ElasticSearchIndexer.add_bulk` API

        """
        config = super().default_configs()
        config.update({
            "fields": ["doc_id", "content"],
            "indexer": {
                "name": "ElasticSearchIndexer",
                "hparams": ElasticSearchIndexer.default_configs(),
                "other_kwargs": {
                    "request_timeout": 10,
                    "refresh": False
                }
            }
        })
        return config

    def _bulk_process(self):
        self.indexer.add_bulk(self.documents,
                              **self.configs.indexer.other_kwargs)


class ElasticSearchTextIndexProcessor(ElasticSearchIndexerBase):
    r"""This processor indexes the text of data packs into an
      `Elasticsearch` index."""

    def _content_for_index(self, input_pack: DataPack) -> List[str]:
        """
        Index two fields, the pack id and the input pack text.

        Args:
            input_pack:

        Returns:

        """
        return [str(input_pack.pack_id), input_pack.text]

    def _field_names(self) -> List[str]:
        return ["doc_id", "content"]


class ElasticSearchPackIndexProcessor(ElasticSearchIndexerBase):
    r"""This processor indexes the data packs into an `Elasticsearch` index."""

    def _content_for_index(self, input_pack: DataPack) -> List[str]:
        """
        Index 3 fields, the pack id, the input pack text and the
          raw pack content.

        Args:
            input_pack:

        Returns:

        """
        return [str(input_pack.pack_id), input_pack.text,
                input_pack.serialize(True)]

    def _field_names(self) -> List[str]:
        return ["doc_id", "content", "pack_info"]
