import os
import logging
import pickle
from abc import ABC
from typing import Optional, Dict, Union
import numpy as np

import torch
from texar.torch import HParams

import faiss

__all__ = [
    "ElasticSearchIndexer",
    "EmbeddingBasedIndexer"
]


class BaseIndexer(ABC):
    r"""Base indexer class. All indexer classes will inherit from this base
    class."""

    def __init__(self):
        pass

    def index(self):
        raise NotImplementedError


class ElasticSearchIndexer(BaseIndexer):
    r"""Indexer class for Elastic Search."""

    def __init__(self):
        super().__init__()
        pass

    def index(self):
        pass


class EmbeddingBasedIndexer(BaseIndexer):
    r"""This class is used for indexing documents represented as vectors. For
    example, each document can be passed through a neural embedding models and
    the vectors are indexed using this class.

    Args:
        hparams (HParams): optional
            Hyperparameters. Missing hyperparameter will be set to default
            values. See :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    INDEX_TYPE_TO_CLASS = {
        "IndexFlatL2": faiss.IndexFlatL2,
        "GpuIndexFlatIP": faiss.GpuIndexFlatIP
    }

    def __init__(self, hparams: Optional[Union[Dict, HParams]] = None):
        super().__init__()
        self._hparams = HParams(hparams=hparams,
                                default_hparams=self.default_hparams())
        self._meta_data: Dict[int, str] = {}

        index_type = self._hparams.index_type
        self.index_class = self.INDEX_TYPE_TO_CLASS[index_type] if \
            isinstance(index_type, str) else index_type

        dim = self._hparams.dim

        # GPU based indexers are prefixed with "Gpu"
        if self.index_class.__name__.startswith("Gpu"):
            res = faiss.StandardGpuResources()
            self._index = self.index_class(res, dim)

        else:
            self._index = self.index_class(dim)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                "index_type": "GpuIndexFlatIP",
                "dim": 768
            }

        Here:

        `"index_type"`: str or class name
            A string or class name representing the index type

            Each line contains a single scalar number.

        `"dim"`: int
            The dimensionality of the vectors that will be indexed.

        """

        return {
            "index_type": "GpuIndexFlatIP",
            "dim": 768
        }

    def index(self):
        pass

    def add(self, vectors: Union[np.ndarray, torch.Tensor],
            meta_data: Dict[int, str]) -> None:
        r"""Add ``vectors`` along with their ``meta_data`` into the index data
        structure.

        Args:
            vectors (np.ndarray or torch.Tensor): A pytorch tensor or a numpy
                array of shape ``[batch_size, *]``.
            meta_data (optional dict): Meta data containing associated with the
                vectors to be added.
        """

        if isinstance(vectors, torch.Tensor):
            # todo: manage the copying of tensors between CPU and GPU
            # efficiently
            vectors = vectors.cpu().numpy()

        self._index.add(vectors)

        self._meta_data.update(meta_data)

    def embed(self):
        pass

    def search(self, query, k):
        r"""Search ``k`` nearest vectors for the ``query`` in the index.

        Args:
            query (numpy array): A 2-dimensional numpy array of shape
                ``[batch_size, dim]``where each row corresponds to a query.
            k (int): An integer representing the number of nearest vectors to
                return from the index

        Returns:
            A list of len ``batch_size`` containing a list of len ``k`` of
            2-D tuples ``(id, meta_data[id])`` containing the id and
            meta-data associated with the vectors.

            .. code-block:: python

                results = index.search(query, k=2)

                # results contains the following
                # [[(id1, txt1)], [(id2, txt2)]]
        """

        _, indices = self._index.search(query, k)
        return [[(idx, self._meta_data[idx]) for idx in index]
                for index in indices]

    def save(self, path: str):
        r"""Save the index and meta data in ``path`` directory. The index
        will be saved as ``index.faiss`` and ``index.meta_data`` respectively
        inside ``path`` directory.

        Args:
            path (str): A path to the directory where the index will be saved

        """

        if os.path.exists(path):
            logging.warning("%s directory already exists. Index will be "
                            "saved into an existing directory", path)
        else:
            os.makedirs(path)

        cpu_index = faiss.index_gpu_to_cpu(self._index)
        faiss.write_index(cpu_index, f"{path}/index.faiss")
        with open(f"{path}/index.meta_data", "wb") as f:
            pickle.dump(self._meta_data, f)

    def load(self, path: str):
        r"""Load the index and meta data from ``path`` directory.

        Args:
            path (str): A path to the directory to load the index from.

        """

        if not os.path.exists(path):
            raise ValueError(f"Failed to load the index. "
                             f"{path} does not exist.")

        # todo: handle the transfer of model between GPU and CPU efficiently
        cpu_index = faiss.read_index(f"{path}/index.faiss")

        if self.index_class.__name__.startswith("Gpu"):
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        else:
            self._index = cpu_index

        with open(f"{path}/index.meta_data", "rb") as f:
            self._meta_data = pickle.load(f)
