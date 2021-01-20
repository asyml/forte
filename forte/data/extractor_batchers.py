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

from abc import abstractmethod
from typing import (
    Dict, List, Iterable, Union, Optional, Tuple, Type, Generic, Iterator, Any)

from forte.common.configuration import Config
from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.types import DataRequest
from forte.data.data_utils_io import merge_batches, batch_instances
from forte.data.ontology.top import Annotation
from forte.data.ontology.core import Entry
from forte.data.converter import Feature

# __all__ = [
#     "ProcessingBatcher",
#     "FixedSizeDataPackBatcher",
#     "FixedSizeMultiPackProcessingBatcher",
# ]


class ProcessingBatcher(Generic[PackType]):
    r"""This defines the basis interface of the Batcher used in
    :class:`~forte.processors.base.batch_processor.BatchProcessor`. This Batcher
    only batches data sequentially. It receives new packs dynamically and cache
    the current packs so that the processors can pack prediction results into
    the data packs.

    Args:
        cross_pack (bool, optional): whether to allow batches go across
        data packs when there is no enough data at the end.
    """

    def __init__(self, cross_pack: bool = True):
        self.pack_pool: List[DataPack] = []
        self.instance_pool: List[Annotation] = []
        self.feature_pool: List[Dict[str, Feature]] = []
        self.pool_size = 0

        self.cross_pack: bool = cross_pack

    def initialize(self, config):
        r"""The implementation should initialize the batcher and setup the
        internal states of this batcher.
        This batcher will be called at the pipeline initialize stage.

        Returns:

        """
        self.scope = config.scope
        self.feature_scheme = config.feature_scheme
        self.pack_pool.clear()
        self.instance_pool.clear()
        self.feature_pool.clear()
        self.pool_size = 0

    @abstractmethod
    def _should_yield(self) -> bool:
        r"""User should implement this based on the state of the batcher to
        indicate whether the batch criteria is met and the batcher should yield
        the current batch. For example, whether the number of instances reaches
        the batch size.

        Returns:

        """
        raise NotImplementedError

    def convert(self, features_collection):
        collections = {}
        for features in features_collection:
            for tag, feat in features.items():
                if tag not in collections:
                    collections[tag] = []
                collections[tag].append(feat)

        converted = {}
        for tag, features in collections.items():
            converter = self.feature_scheme[tag]["converter"]
            data, masks = converter.convert(features)
            converted[tag]["data"] = data
            converted[tag]["mask"] = masks
            converted[tag]["features"] = features
        return converted

    def flush(self) -> Iterator[Dict]:
        r"""Flush the remaining data.

        Returns:

        """
        if len(self.feature_pool) > 0:
            yield (self.pack_pool, self.instance_pool,
                    self.convert(self.feature_pool))
            self.feature_pool.clear()
            self.pack_pool.clear()
            self.instance_pool.clear()
            self.pool_size = 0

    def get_batch(
            self, input_pack: PackType) -> Iterator[Dict]:
        r"""Returns an iterator of data batches."""
        # cache the new pack and generate batches

        for (batch, num) in self._get_data_batch(input_pack):
            packs, instance, features = batch
            self.pack_pool.extend(packs)
            self.instance_pool.extend(instance)
            self.feature_pool.extend(features)
            self.pool_size += num

            # Yield a batch on two conditions.
            # 1. If we do not want to have batches from different pack, we
            # should yield since this pack is exhausted.
            # 2. We should also yield when the batcher condition is met:
            # i.e. ``_should_yield()`` is True.
            if not self.cross_pack or self._should_yield():
                self.flush()

    def _get_data_batch(
            self, data_pack: PackType) -> Iterable[Tuple[Dict, Annotation, int]]:
        r"""Get data batches based on the requests.

        Args:
            data_pack: The data pack to retrieve data from.
            context_type: The context type of the data pack.
            requests: The request detail.
            offset: The offset for get_data.

        Returns:

        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "scope": None,
            "feature_scheme": {}
        }


class FixedSizeDataPackBatcher(ProcessingBatcher[DataPack]):
    def initialize(self, config: Config):
        super().initialize(config)
        self.batch_size = config.batch_size
        self.batch_is_full = False

    def _should_yield(self) -> bool:
        return self.batch_is_full

    def _get_data_batch(self, data_pack: DataPack) -> Iterable[Tuple[Dict, int]]:
        r"""Try to get batches from a dataset  with ``batch_size``, but will
        yield an incomplete batch if the data_pack is exhausted.

        Returns:
            An iterator of tuples ``(batch, cnt)``, ``batch`` is a dict
            containing the required annotations and context, and ``cnt`` is
            the number of instances in the batch.
        """
        packs: List[DataPack] = []
        instances: List[Annotation] = []
        features_collection: List[Dict[str, Feature]] = []
        current_size = self.pool_size

        for instance in data_pack.get(self.scope):
            features = {}
            for tag, scheme in self.feature_scheme.items():
                features[tag] = scheme['extractor'].extract(data_pack, instance)
            packs.append(data_pack)
            instances.append(instance)
            features_collection.append(features)

            if len(instances) == self.batch_size - current_size:
                self.batch_is_full = True
                batch = (packs, instances, features_collection)
                yield (batch, len(instances))
                self.batch_is_full = False
                packs.clear()
                instances.clear()
                features_collection.clear()
                current_size = self.pool_size

        # Flush the remaining data.
        if len(instances) > 0:
            batch = (packs, instances, features_collection)
            yield (batch, len(instances))

    @classmethod
    def default_configs(cls) -> Dict:
        config = super().default_configs()
        config.update({
            "batch_size": 10
        })
        return config
