"""This module tests indexer module."""
import unittest
import numpy as np

from forte.indexers import EmbeddingBasedIndexer


class TestEmbeddingBasedIndexer(unittest.TestCase):
    r"""Tests Embedding based indexers."""

    def setUp(self):
        self.index = EmbeddingBasedIndexer(
            hparams={"index_type": "IndexFlatL2", "dim": 2})

        vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
        self.index.add(vectors, meta_data={0: "0", 1: "1", 2: "2"})

    def test_indexer(self):
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
