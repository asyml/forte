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

from copy import deepcopy
from typing import Optional, Dict, Union, Any, Iterable

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

__all__ = [
    "ElasticSearchIndexer"
]

from forte.common.configuration import Config


class ElasticSearchIndexer:
    r"""Indexer class for Elastic Search."""

    def __init__(self, config: Optional[Union[Dict, Config]] = None):
        super().__init__()
        self._config = Config(config, self.default_configs())
        self.elasticsearch = Elasticsearch(hosts=self._config.hosts)

    def index(self, document: Dict[str, Any], index_name: Optional[str] = None,
              refresh: Optional[Union[bool, str]] = False) -> None:
        r"""Index a document ``document`` in the index specified by
        ``index_name``. If ``index_name`` is None, it will be picked from
        hparams.

        Args:
            document (Dict): Document to be indexed into Elasticsearch indexer
            index_name (str): Name of the index where this document will be
                saved. If None, value will be picked from hparams.
            refresh (bool, str): refresh settings to control when changes
                made by this request are made visible to search. Available
                value are "True","wait_for", "False"

            .. note::
                "refresh" setting will greatly affect the Elasticsearch
                performance. Please refer to
                https://www.elastic.co/guide/en/elasticsearch/reference/master/docs-refresh.html
                for more information on "refresh"
        """
        self.add(document, index_name, refresh)

    def add(self, document: Dict[str, Any], index_name: Optional[str] = None,
            refresh: Optional[Union[bool, str]] = False) -> None:
        r"""Add a document ``document`` to the index specified by
        ``index_name``. If ``index_name`` is None, it will be picked from
        hparams.

        Args:
            document (Dict): Document to be indexed into Elasticsearch indexer
            index_name (str): Name of the index where this document will be
                saved. If None, value will be picked from hparams.
            refresh (bool, str): refresh settings to control when changes
                made by this request are made visible to search. Available
                value are "True","wait_for", "False"

            .. note::
                "refresh" setting will greatly affect the Elasticsearch
                performance. Please refer to
                https://www.elastic.co/guide/en/elasticsearch/reference/master/docs-refresh.html
                for more information on "refresh"
        """
        index_name = index_name if index_name else self._config.index_name
        self.elasticsearch.index(  # pylint: disable=unexpected-keyword-arg
            index=index_name, body=document, refresh=refresh)

    def add_bulk(self, documents: Iterable[Dict[str, Any]],
                 index_name: Optional[str] = None,
                 **kwargs: Optional[Dict[str, Any]]) -> None:
        r"""Add a bulk of documents to the index specified by ``index_name``.
        If ``index_name`` is None, it will be picked from hparams.

        Args:
            documents (Iterable): An iterable of documents to be indexed.
            index_name (optional, str): Name of the index where this document
                will be saved. If None, value will be picked from hparams.
            kwargs (optional, dict) : Optional keyword arguments like
                "refresh", "request_timeout" etc. that are passed to
                Elasticsearch's bulk API. Please refer to
                https://elasticsearch-py.readthedocs.io/en/master/helpers.html#bulk-helpers
                for the complete list of arguments.

            .. note::
                "refresh" setting will greatly affect the Elasticsearch
                performance. Please refer to
                https://www.elastic.co/guide/en/elasticsearch/reference/master/docs-refresh.html
                for more information on "refresh"
        """

        def actions():
            for document in documents:
                new_document = deepcopy(document)
                new_document.update(
                    {"_index": index_name if index_name else
                    self.hparams.index_name,
                     "_type": "document"})
                yield new_document

        bulk(self.elasticsearch, actions(), **kwargs)

    def search(self, query: Dict[str, Any], index_name: Optional[str] = None,
               **kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        r"""Search the index specified by ``index_name`` that matches the
        ``query``.

        Args:
             query (dict): An elasticsearch query which is issued to the indexer
             index_name (str): Name of the index where documents are looked up.
                If None, value will be picked from hparams.
             kwargs (optional, dict) : Optional keyword arguments like
                "size", "request_timeout" etc. that are passed to
                Elasticsearch's bulk API. Please refer to
                https://elasticsearch-py.readthedocs.io/en/master/api.html#elasticsearch.Elasticsearch.search
                for the complete list of arguments.

        Returns:
            A dict containing the documents matching the query along with
            meta data of the search.
        """
        index_name = index_name if index_name else self.hparams.index_name
        return self.elasticsearch.search(index=index_name, body=query, **kwargs)

    @property
    def hparams(self):
        return self._config

    @staticmethod
    def default_configs() -> Dict[str, Any]:
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                "index_name": "elastic_indexer",
                "hosts": "localhost:9200",
                "algorithm": "bm25"
            }

        Here:

        `"index_name"`: str
            A string representing the index to which the documents will be
            added.

        `"hosts"`: list, str
            A list of hosts or a host which the Elasticsearch client will be
            connected to.

        """
        return {
            "index_name": "elastic_indexer",
            "hosts": "localhost:9200",
            "algorithm": "bm25"
        }
