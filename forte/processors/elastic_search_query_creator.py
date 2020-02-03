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
from typing import Any, Dict, Tuple

from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data import DataPack, MultiPack
from forte.processors.base import QueryProcessor

__all__ = [
    "ElasticSearchQueryCreator"
]


class ElasticSearchQueryCreator(QueryProcessor[MultiPack]):
    r"""This processor creates a Elasticsearch query and adds it as Query entry
    in the data pack. This query will later used by a Search processor to
    retrieve documents."""

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, resources: Resources, configs: HParams):
        self.resource = resources
        self.config = configs

    def _build_query(self, text: str) -> Dict[str, Any]:
        r"""Constructs Elasticsearch query that will be consumed by
        Elasticsearch processor.

        Args:
             text: str
                A string which will be looked up for in the corpus under field
                name `field`. `field` can be passed in a `config` during
                :meth:`ElasticSearchQueryCreator::initialize`. If `config` does
                not contain the key `field`, we will set it to "content"
        """
        size = self.config.size or 1000
        field = self.config.field or "content"
        return {"query": {"match": {field: text}}, "size": size}

    @staticmethod
    def default_configs() -> Dict[str, Any]:
        return {
            "size": 1000,
            "field": "content",
            "query_pack_name": "query"
        }

    def _process_query(self, input_pack: MultiPack) -> \
            Tuple[DataPack, Dict[str, Any]]:
        query_pack = input_pack.get_pack(self.config.query_pack_name)
        query = self._build_query(text=query_pack.text)
        return query_pack, query
