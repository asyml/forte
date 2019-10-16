import os
import logging
import pickle
from abc import ABC
from typing import Optional, Dict, Union

import faiss

from texar.torch import HParams

__all__ = [
    "ElasticSearchIndexer",
    "EmbeddingBasedIndexer"
]


class BaseIndexer(ABC):
    def __init__(self):
        pass

    def index(self):
        raise NotImplementedError


class ElasticSearchIndexer(BaseIndexer):
    def __init__(self):
        super().__init__()
        pass

    def index(self):
        pass


class EmbeddingBasedIndexer(BaseIndexer):
    def __init__(self, hparams: Optional[Union[Dict, HParams]] = None):
        super().__init__()
        self._hparams = HParams(hparams=hparams,
                                default_hparams=self.default_hparams())
        self._meta_data: Dict[int, str] = {}
        res = faiss.StandardGpuResources()
        dim = self._hparams.dim
        self._index = faiss.GpuIndexFlatIP(res, dim)

    def default_hparams(self):  # pylint: disable=no-self-use
        return {
            "index_type": "GpuIndexFlatIP",
            "dim": 768
        }

    def index(self):
        pass

    def add(self, vectors, meta_data):
        # todo: manage the copying of tensors between CPU and GPU efficiently
        self._index.add(vectors.cpu().numpy())
        self._meta_data.update(meta_data)

    def embed(self):
        pass

    def search(self, query, k):
        r"""Search ``k`` nearest vectors for the ``query`` in the index.

        Args:
            query (numpy array): A 2-dimensional numpy array where each row
                corresponds to a query. ``query.shape[1]`` is the dimensionality
                of the embedding
            k (int): An integer representing the number of nearest vectors to
                return from the index

        Returns:

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
            logging.warning("f{path} directory already exists. Index will be "
                            "saved into an existing directory")
        else:
            os.makedirs(name=path)

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
        res = faiss.StandardGpuResources()
        self._index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

        with open(f"{path}/index.meta_data", "rb") as f:
            self._meta_data = pickle.load(f)


class BertBasedIndexer(BaseIndexer):
    def __init__(self):
        super().__init__()
        pass

    def index(self):
        pass

    def embed(self):
        pass
