# Copyright 2020 The Forte Authors. All Rights Reserved.
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
from forte.processors.base.pack_processor import PackProcessor

__all__ = [
    "IndexProcessor",
    "IndexProcessorWithDatapack"
]


class IndexProcessor(PackProcessor, ABC):
    r"""A  base processor for indexing documents into traditional indexers like
    Elasticsearch and/or dense vector indexers like Faiss. Subclasses need to
    implement :meth:`IndexProcessor::_bulk_process`.

    """

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()
        # self.documents: List[List[str]] = []
        self.documents: List[Dict[str, str]] = []

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

    def _field_names(self) -> List[str]:
        r"""Subclasses of :class:`IndexProcessor` should implement this method
          to provide the field name for indexing.
          The return value from :func:`_content_for_index` will be added into
          these fields. The length of the return value of this function should
          be the same as the return value for :func:`_content_for_index`.
        Returns:

        """
        raise NotImplementedError

    def _content_for_index(self, input_pack: DataPack) -> List[str]:
        raise NotImplementedError

    def _process(self, input_pack: DataPack):
        # self.documents.append((str(input_pack.pack_id), input_pack.text))
        index_pairs: Dict[str, str] = dict(
            zip(self._field_names(), self._content_for_index(input_pack)))
        self.documents.append(index_pairs)

        if len(self.documents) == self.configs.batch_size:
            self._bulk_process()
            self.documents = []

    def flush(self):
        if len(self.documents) > 0:
            self._bulk_process()


class IndexProcessorWithDatapack(PackProcessor, ABC):
    r"""A  base processor for indexing a document with its original datapack
    into traditional indexers like Elasticsearch.
    Subclasses need to implement
    :meth:`IndexProcessorWithDatapack::_bulk_process`.
    """

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()
        self.documents: List[Tuple[str, str, str]] = []

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
        r"""Subclasses of :class:`IndexProcessorWithDatapack`
        should implement this method
        to bulk add the documents into the index.
        """
        raise NotImplementedError

    def _process(self, input_pack: DataPack):
        serialized_datapack: str = input_pack.serialize()

        self.documents.append((str(input_pack.pack_id), input_pack.text,
                               serialized_datapack))

        if len(self.documents) == self.config.batch_size:
            self._bulk_process()
            self.documents = []

    def flush(self):
        if len(self.documents) > 0:
            self._bulk_process()
