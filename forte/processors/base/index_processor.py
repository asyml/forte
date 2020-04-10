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
Index processor
"""
from abc import ABC
from typing import Dict, Any, List, Tuple

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
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

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, resources: Resources, configs: Config):
        self.resources = resources
        self.config = configs

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "batch_size": 128
        })
        return config

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
