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
from abc import ABC
from typing import Dict, Any, List, Tuple

from texar.torch import HParams
from forte.common import Resources
from forte.data import DataPack
from forte.processors.base.base_processor import BaseProcessor

__all__ = [
    "IndexProcessor"
]


class IndexProcessor(BaseProcessor[DataPack], ABC):
    r"""A  base processor for indexing documents into traditional indexers like
    Elasticsearch and/or dense vector indexers like Faiss. Subclasses need to
    implement :meth:`IndexProcessor::_bulk_process`.

    """

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()
        self.documents: List[Tuple[str, str]] = []

    def initialize(self, resources: Resources, configs: HParams):
        self.resources = resources
        self.config = configs

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        return {
            "batch_size": 128
        }

    def _bulk_process(self):
        r"""Subclasses of :class:`IndexProcessor` should implement this method
        to bulk add the documents into the index.
        """
        raise NotImplementedError

    def _process(self, input_pack: DataPack):
        if input_pack.meta.doc_id:
            self.documents.append((input_pack.meta.doc_id, input_pack.text))
        else:
            self.documents.append(("DOC", input_pack.text))

        if len(self.documents) == self.config.batch_size:
            self._bulk_process()
            self.documents = []

    def flush(self):
        if len(self.documents) > 0:
            self._bulk_process()
