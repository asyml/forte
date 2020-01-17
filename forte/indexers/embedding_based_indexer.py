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
import os
import logging
import pickle
from typing import Optional, List, Tuple, Dict, Union, Any
import numpy as np

import torch
from texar.torch import HParams
import faiss

from forte import utils

__all__ = [
    "EmbeddingBasedIndexer"
]


class EmbeddingBasedIndexer:
    r"""This class is used for indexing documents represented as vectors. For
    example, each document can be passed through a neural embedding models and
    the vectors are indexed using this class.

    Args:
        hparams (HParams): optional
            Hyperparameters. Missing hyperparameter will be set to default
            values. See :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    INDEX_TYPE_TO_CONFIG = {
        "GpuIndexFlatL2": "GpuIndexFlatConfig",
        "GpuIndexFlatIP": "GpuIndexFlatConfig",
        "GpuIndexIVFFlat": "GpuIndexIVFFlatConfig"
    }

    def __init__(self, hparams: Optional[Union[Dict, HParams]] = None):
        super().__init__()
        self._hparams = HParams(hparams=hparams,
                                default_hparams=self.default_hparams())
        self._meta_data: Dict[int, str] = {}

        index_type = self._hparams.index_type
        device = self._hparams.device
        dim = self._hparams.dim

        if device.lower().startswith("gpu"):
            if isinstance(index_type, str) and not index_type.startswith("Gpu"):
                index_type = "Gpu" + index_type

            index_class = utils.get_class(index_type, module_paths=["faiss"])
            gpu_resource = faiss.StandardGpuResources()
            gpu_id = int(device[3:])
            if faiss.get_num_gpus() < gpu_id:
                gpu_id = 0
                logging.warning("Cannot create the index on device %s. "
                                "Total number of GPUs on this machine is "
                                "%s. Using gpu0 for the index.",
                                self._hparams.device, faiss.get_num_gpus())
            config_class_name = \
                self.INDEX_TYPE_TO_CONFIG.get(index_class.__name__)
            config = utils.get_class(config_class_name,  # type: ignore
                                     module_paths=["faiss"])()
            config.device = gpu_id
            self._index = index_class(gpu_resource, dim, config)

        else:
            index_class = utils.get_class(index_type, module_paths=["faiss"])
            self._index = index_class(dim)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                "index_type": "IndexFlatIP",
                "dim": 768,
                "device": "cpu"
            }

        Here:

        `"index_type"`: str or class name
            A string or class name representing the index type

            Each line contains a single scalar number.

        `"dim"`: int
            The dimensionality of the vectors that will be indexed.

        """

        return {
            "index_type": "IndexFlatIP",
            "dim": 768,
            "device": "cpu"
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
            meta_data (optional dict): Meta data associated with the vectors to
                be added. Meta data can include the document contents of the
                vectors.
        """

        if isinstance(vectors, torch.Tensor):
            # todo: manage the copying of tensors between CPU and GPU
            # efficiently
            vectors = vectors.cpu().numpy()

        self._index.add(vectors)

        self._meta_data.update(meta_data)

    def embed(self):
        pass

    def search(self, query: np.ndarray, k: int) -> List[List[Tuple[int, str]]]:
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

    def save(self, path: str) -> None:
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

        cpu_index = faiss.index_gpu_to_cpu(self._index) \
            if self._index.__class__.__name__.startswith("Gpu") else self._index
        faiss.write_index(cpu_index, f"{path}/index.faiss")
        with open(f"{path}/index.meta_data", "wb") as f:
            pickle.dump(self._meta_data, f)

    def load(self, path: str, device: Optional[str] = None) -> None:
        r"""Load the index and meta data from ``path`` directory.

        Args:
            path (str): A path to the directory to load the index from.
            device (optional str): Device to load the index into. If None,
                value will be picked from hyperparameters.

        """

        if not os.path.exists(path):
            raise ValueError(f"Failed to load the index. {path} "
                             f"does not exist.")

        cpu_index = faiss.read_index(f"{path}/index.faiss")

        if device is None:
            device = self._hparams.device

        if device.lower().startswith("gpu"):
            gpu_resource = faiss.StandardGpuResources()
            gpu_id = int(device[3:])
            if faiss.get_num_gpus() < gpu_id:
                gpu_id = 0
                logging.warning("Cannot create the index on device %s. "
                                "Total number of GPUs on this machine is "
                                "%s. Using the gpu0 for the index.",
                                device, faiss.get_num_gpus())
            self._index = faiss.index_cpu_to_gpu(
                gpu_resource, gpu_id, cpu_index)

        else:
            self._index = cpu_index

        with open(f"{path}/index.meta_data", "rb") as f:
            self._meta_data = pickle.load(f)
