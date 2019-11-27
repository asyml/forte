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
from typing import Union, Tuple
import numpy as np

from forte.data.base_pack import PackType
from forte.data.ontology import Query
from forte.processors.base.pack_processor import BasePackProcessor

__all__ = [
    "QueryProcessor"
]

QueryType = Union[str, np.ndarray]


class QueryProcessor(BasePackProcessor[PackType], ABC):
    r"""A base class for all processors that handle query creation for
    information retrieval."""

    def _build_query(self, text: str) -> QueryType:
        r"""Subclasses of QueryProcessor need to implement this method to create
        a query from a text string.

        Args:
            text (str): A string for which a query will be generated

        Returns:
            A str or numpy array representing a query for `text`

        """
        raise NotImplementedError

    def _process_query(self, input_pack: PackType) \
            -> Tuple[PackType, QueryType]:
        r"""Subclasses of QueryProcessor should implement this method which
        takes in an `input_pack` and processes it to generate a query.

        Args:
            input_pack (DataPack or a MultiPack): A (data/multi)-pack for which
                a query is generated. If `input_pack` is a multipack, the
                processor should fetch the relevant datapack and process it to
                generate a query.

        Returns:
             A tuple containing the `(query_pack, query)`.

             - If `input_pack` is a multipack, `query_pack` is one of its data
             pack for which a `query` is generated.
             - If `input_pack` is a datapack, `query_pack` is the `input_pack`.
        """
        raise NotImplementedError

    def _process(self, input_pack: PackType):
        query_pack, query_vector = self._process_query(input_pack)
        query = Query(pack=query_pack, value=query_vector)
        query_pack.add_or_get_entry(query)
