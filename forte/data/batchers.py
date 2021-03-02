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
from torch import Tensor
from forte.common.configuration import Config
from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.types import DataRequest
from forte.data.data_utils_io import merge_batches, batch_instances
from forte.data.ontology.top import Annotation
from forte.data.ontology.core import Entry, EntryType
from forte.data.converter import Feature

__all__ = [
    "ProcessingBatcher",
    "FixedSizeDataPackBatcherWithExtractor",
    "FixedSizeDataPackBatcher",
    "FixedSizeMultiPackProcessingBatcher",
]


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
        self.current_batch: Dict = {}
        self.data_pack_pool: List[PackType] = []
        self.current_batch_sources: List[int] = []

        self.cross_pack: bool = cross_pack

    def initialize(self, _):
        r"""The implementation should initialize the batcher and setup the
        internal states of this batcher.
        This batcher will be called at the pipeline initialize stage.

        Returns:

        """
        self.current_batch.clear()
        self.data_pack_pool.clear()
        self.current_batch_sources.clear()

    @abstractmethod
    def _should_yield(self) -> bool:
        r"""User should implement this based on the state of the batcher to
        indicate whether the batch criteria is met and the batcher should yield
        the current batch. For example, whether the number of instances reaches
        the batch size.

        Returns:

        """
        raise NotImplementedError

    def flush(self) -> Iterator[Tuple[List[PackType],
                    Optional[List[Annotation]], Dict]]:
        r"""Flush the remaining data.

        Returns:
            A tuple contains datapack, instance and batch data.
            In the basic ProcessingBatcher, to be compatible with
            existing implementation, instance is not needed, thus
            using None.
        """
        if self.current_batch:
            yield self.data_pack_pool, None, self.current_batch
            self.current_batch = {}
            self.current_batch_sources = []
            self.data_pack_pool = []

    def get_batch(
            self, input_pack: PackType, context_type: Type[Annotation],
            requests: DataRequest):
        r"""Returns an iterator of A tuple contains datapack,
        instance and batch data. In the basic ProcessingBatcher,
        to be compatible with existing implementation,
        instance is not needed, thus using None."""
        # cache the new pack and generate batches

        for (data_batch, instance_num) in self._get_data_batch(
                input_pack, context_type, requests):
            self.current_batch = merge_batches(
                [self.current_batch, data_batch])
            self.current_batch_sources.append(instance_num)
            self.data_pack_pool.extend([input_pack] * instance_num)

            # Yield a batch on two conditions.
            # 1. If we do not want to have batches from different pack, we
            # should yield since this pack is exhausted.
            # 2. We should also yield when the batcher condition is met:
            # i.e. ``_should_yield()`` is True.
            if not self.cross_pack or self._should_yield():
                yield self.data_pack_pool, None, self.current_batch
                self.current_batch = {}
                self.current_batch_sources = []
                self.data_pack_pool = []

    def _get_data_batch(
            self, data_pack: PackType, context_type: Type[Annotation],
            requests: Optional[DataRequest] = None, offset: int = 0) \
            -> Iterable[Tuple[Dict, int]]:
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
    @abstractmethod
    def default_configs(cls) -> Dict[str, Any]:
        raise NotImplementedError


class FixedSizeDataPackBatcherWithExtractor(ProcessingBatcher):
    r"""This class use extractor to extract features from
    dataset and group them into batch. In this class, more pools
    are added. One is `instance_pool`, which is used to record the
    instance from which feature is extracted. The other one is
    `feature_pool`, which is used to record features before they
    can be yeild in batch.

    Args:
        cross_pack (bool, optional): whether to allow batches go across
        data packs when there is no enough data at the end.
    """
    def __init__(self, cross_pack: bool = True):
        super().__init__(cross_pack=cross_pack)
        self.instance_pool: List[Annotation] = []
        self.feature_pool: List[Dict[str, Feature]] = []
        self.pool_size = 0
        self.batch_is_full = False

    def initialize(self, config):
        super().initialize(config)

        if config["scope"] is None or \
            config["feature_scheme"] is None or \
            config["batch_size"] is None:
            raise AttributeError("scope, feature_scheme and"
                                "batch_size cannot be None"
                                "in the config.")
        self.scope: Type[EntryType] = config["scope"]
        self.feature_scheme: Dict = config["feature_scheme"]
        self.batch_size: int = config["batch_size"]
        self.instance_pool.clear()
        self.feature_pool.clear()
        self.pool_size = 0
        self.batch_is_full = False

    def convert(self, features_collection) -> \
                Dict[str, Union[Tensor, Dict]]:
        r"""This function use converter to turn a
        list of features into batch.

        Args:
            features_collectioin (List[Dict[str, Feature]]):
                A list of features.

        Returns:
            A instance of Dict[str, Union[Tensor, Dict]], which
            is a batch of features.
        """
        collections: Dict[str, List[Feature]] = {}
        for features in features_collection:
            for tag, feat in features.items():
                if tag not in collections:
                    collections[tag] = []
                collections[tag].append(feat)

        converted = {}
        for tag, features in collections.items():
            converter = self.feature_scheme[tag]["converter"]
            data, masks = converter.convert(features)
            converted[tag] = {
                "data": data,
                "masks": masks,
                "features": features
            }
        return converted

    def _should_yield(self) -> bool:
        return self.batch_is_full

    def flush(self):
        r"""Flush the remaining data."""
        if self.pool_size > 0:
            yield (self.data_pack_pool, self.instance_pool,
                    self.convert(self.feature_pool))
            self.data_pack_pool = []
            self.instance_pool = []
            self.feature_pool = []
            self.pool_size = 0

    def get_batch(
            self, input_pack: PackType, context_type: Type[Annotation],
            requests: DataRequest) -> Iterator[Tuple[List[PackType],
                    Optional[List[Annotation]], Dict]]:
        r"""Returns an iterator of data batches."""
        # cache the new pack and generate batches

        for (batch, num) in self._get_data_batch(input_pack, context_type):
            packs, instance, features = batch["dummy"]
            self.data_pack_pool.extend(packs)
            self.instance_pool.extend(instance)
            self.feature_pool.extend(features)
            self.pool_size += num

            # Yield a batch on two conditions.
            # 1. If we do not want to have batches from different pack, we
            # should yield since this pack is exhausted.
            # 2. We should also yield when the batcher condition is met:
            # i.e. ``_should_yield()`` is True.
            if not self.cross_pack or self._should_yield():
                yield from self.flush()

    def _get_data_batch(
            self, data_pack: PackType, context_type: Type[Annotation],
            requests: Optional[DataRequest] = None, offset: int = 0) \
            -> Iterable[Tuple[Dict[Any, Any], int]]:
        r"""Get data batches based on the requests.

        Args:
            data_pack: The data pack to retrieve data from.
            context_type: The context type of the data pack.
                This is not used and is only for compatibility reason.
            requests: The request detail.
                This is not used and is only for compatiblilty reason.
            offset: The offset for get_data.
                This is not used and is only for compatibility reason.
        """
        packs: List[PackType] = []
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
                batch = {"dummy": (packs, instances, features_collection)}
                yield (batch, len(instances))
                self.batch_is_full = False
                packs = []
                instances = []
                features_collection = []
                current_size = self.pool_size

        # Flush the remaining data.
        if len(instances) > 0:
            batch = {"dummy": (packs, instances, features_collection)}
            yield (batch, len(instances))

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = {
            "scope": None,
            "feature_scheme": None,
            "batch_size": None
        }
        return config


class FixedSizeDataPackBatcher(ProcessingBatcher[DataPack]):
    def initialize(self, config: Config):
        super().initialize(config)
        self.batch_size = config.batch_size
        self.batch_is_full = False

    def _should_yield(self) -> bool:
        return self.batch_is_full

    def _get_data_batch(
            self, data_pack: DataPack, context_type: Type[Annotation],
            requests: Optional[Dict[Type[Entry], Union[Dict, List]]] = None,
            offset: int = 0) -> Iterable[Tuple[Dict, int]]:
        r"""Try to get batches from a dataset  with ``batch_size``, but will
        yield an incomplete batch if the data_pack is exhausted.

        Returns:
            An iterator of tuples ``(batch, cnt)``, ``batch`` is a dict
            containing the required annotations and context, and ``cnt`` is
            the number of instances in the batch.
        """
        instances: List[Dict] = []
        current_size = sum(self.current_batch_sources)

        for data in data_pack.get_data(context_type, requests, offset):
            instances.append(data)
            if len(instances) == self.batch_size - current_size:
                batch = batch_instances(instances)
                self.batch_is_full = True
                yield (batch, len(instances))
                instances = []
                self.batch_is_full = False

        # Flush the remaining data.
        if len(instances) > 0:
            batch = batch_instances(instances)
            yield (batch, len(instances))

    @classmethod
    def default_configs(cls) -> Dict:
        return {
            'batch_size': 10
        }


class FixedSizeMultiPackProcessingBatcher(ProcessingBatcher[MultiPack]):
    r"""A Batcher used in ``MultiPackBatchProcessors``.

    The Batcher calls the ProcessingBatcher inherently on each specified
    data pack in the MultiPack.

    It's flexible to query MultiPack so we delegate the task to the subclasses
    such as:
        - query all packs with the same ``context`` and ``input_info``.
        - query different packs with different ``context``s and ``input_info``s.

    Since the batcher will save the data_pack_pool on the fly, it's not trivial
    to do batching and slicing multiple data packs in the same time
    """

    def __init__(self, cross_pack: bool = True):
        super().__init__(cross_pack)
        self.batch_is_full = False

    def initialize(self, config: Config):
        super().initialize(config)
        self.input_pack_name = config.input_pack_name
        self.batch_size = config.batch_size
        self.batch_is_full = False

    def _should_yield(self) -> bool:
        return self.batch_is_full

    # TODO: Principled way of get data from multi pack?
    def _get_data_batch(
            self, multi_pack: MultiPack, context_type: Type[Annotation],
            requests: Optional[Dict[Type[Entry], Union[Dict, List]]] = None,
            offset: int = 0) -> Iterable[Tuple[Dict, int]]:
        r"""Try to get batches of size ``batch_size``. If the tail instances
        cannot make up a full batch, will generate a small batch with the tail
        instances.

        Returns:
            An iterator of tuples ``(batch, cnt)``, ``batch`` is a dict
            containing the required annotations and context, and ``cnt`` is
            the number of instances in the batch.
        """
        input_pack = multi_pack.get_pack(self.input_pack_name)

        instances: List[Dict] = []
        current_size = sum(self.current_batch_sources)
        for data in input_pack.get_data(context_type, requests, offset):
            instances.append(data)
            if len(instances) == self.batch_size - current_size:
                batch = batch_instances(instances)
                self.batch_is_full = True
                yield (batch, len(instances))
                instances = []
                self.batch_is_full = False

        if len(instances):
            batch = batch_instances(instances)
            yield (batch, len(instances))

    @classmethod
    def default_configs(cls) -> Dict:
        return {
            'batch_size': 10,
            'input_pack_name': 'source'
        }
