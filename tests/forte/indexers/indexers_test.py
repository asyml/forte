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
"""
Unit tests for indexer module.
"""

import time
import unittest
from ddt import ddt, data, unpack
import shutil
import os
import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from forte.indexers.embedding_based_indexer import EmbeddingBasedIndexer
from forte.indexers.elastic_indexer import ElasticSearchIndexer
from tests.utils import performance_test


class TestEmbeddingBasedIndexer(unittest.TestCase):
    r"""Tests Embedding based indexers."""

    @classmethod
    def setUpClass(cls):
        cls.index = EmbeddingBasedIndexer(
            hparams={"index_type": "IndexFlatL2", "dim": 2})

        vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        cls.index.add(vectors, meta_data={0: "0", 1: "1", 2: "2"})
        cls.index_path = "index/"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.index_path):
            shutil.rmtree(cls.index_path)

    def test_indexer(self):
        self.assertEqual(self.index._index.ntotal, 3)
        actual_results = [[np.array([1, 0], dtype=np.float32),
                           np.array([1, 1], dtype=np.float32)],
                          [np.array([1, 0], dtype=np.float32),
                           np.array([0, 1], dtype=np.float32)]]
        search_vectors = np.array([[1, 0], [0.25, 0]], dtype=np.float32)
        results = self.index.search(search_vectors, k=2)
        for i, result in enumerate(results):
            for j, t in enumerate(result):
                self.assertTrue(
                    np.all(actual_results[i][j] ==
                           self.index._index.reconstruct(int(t[0]))))

    def test_indexer_save_and_load(self):
        self.index.save(path=self.index_path)
        saved_files = ["index.faiss", "index.meta_data"]
        for file in os.listdir(path=self.index_path):
            self.assertIn(file, saved_files)

        new_index = EmbeddingBasedIndexer(
            hparams={"index_type": "IndexFlatL2", "dim": 2})
        new_index.load(path=self.index_path, device="cpu")

        self.assertEqual(new_index._index.ntotal, 3)
        actual_results = [[np.array([1, 0], dtype=np.float32),
                           np.array([1, 1], dtype=np.float32)],
                          [np.array([1, 0], dtype=np.float32),
                           np.array([0, 1], dtype=np.float32)]]
        search_vectors = np.array([[1, 0], [0.25, 0]], dtype=np.float32)
        results = new_index.search(search_vectors, k=2)
        for i, result in enumerate(results):
            for j, t in enumerate(result):
                self.assertTrue(
                    np.all(actual_results[i][j] ==
                           new_index._index.reconstruct(int(t[0]))))


@ddt
class TestElasticSearchIndexer(unittest.TestCase):
    r"""Tests Elastic Indexer."""

    def setUp(self):
        self.indexer = ElasticSearchIndexer(
            hparams={"index_name": "test_index"})

    def tearDown(self):
        self.indexer.elasticsearch.indices.delete(
            index=self.indexer.hparams.index_name, ignore=[400, 404])

    def test_add(self):
        document = {"key": "This document is created to test "
                           "ElasticSearchIndexer"}
        self.indexer.add(document, refresh="wait_for")
        retrieved_document = self.indexer.search(
            query={"query": {"match": {"key": "ElasticSearchIndexer"}},
                   "_source": ["key"]})
        hits = retrieved_document["hits"]["hits"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["_source"], document)

    def test_add_bulk(self):
        size = 10000
        documents = set([f"This document {i} is created to test "
                         f"ElasticSearchIndexer" for i in range(size)])
        self.indexer.add_bulk([{"key": document} for document in documents],
                              refresh="wait_for")
        retrieved_document = self.indexer.search(
            query={"query": {"match_all": {}}},
            index_name="test_index", size=size)
        hits = retrieved_document["hits"]["hits"]
        self.assertEqual(len(hits), size)
        results = set([hit["_source"]["key"] for hit in hits])
        self.assertEqual(results, documents)

    @performance_test
    @data([100, 0.3], [500, 0.3], [1000, 0.3])
    @unpack
    def test_speed(self, size, epsilon):
        es = Elasticsearch()
        documents = [{"_index": "test_index_",
                      "_type": "document",
                      "key": f"This document {i} is created to test "
                             f"ElasticSearchIndexer"} for i in range(size)]

        start = time.time()
        bulk(es, documents, refresh=False)
        baseline = time.time() - start
        es.indices.delete(index="test_index_", ignore=[400, 404])

        documents = set([f"This document {i} is created to test "
                         f"ElasticSearchIndexer" for i in range(size)])
        start = time.time()
        self.indexer.add_bulk([{"key": document} for document in documents],
                              refresh=False)
        forte_time = time.time() - start
        self.assertLessEqual(forte_time, baseline + epsilon)


if __name__ == '__main__':
    unittest.main()
