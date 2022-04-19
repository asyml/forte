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


from abc import abstractmethod
from typing import (
    Dict,
    List,
    Iterable,
    Union,
    Optional,
    Tuple,
    Generic,
    Iterator,
    Any,
)

from forte.common import ProcessorConfigError
from forte.common.configurable import Configurable
from forte.common.configuration import Config
from forte.data.base_pack import PackType


from forte.data.data_pack import DataPack
from forte.data.data_utils_io import merge_batches, batch_instances
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Annotation


__all__ = [
    "ProcessingBatcher",
    "FixedSizeRequestDataPackBatcher",
    "FixedSizeMultiPackProcessingBatcher",
    "FixedSizeDataPackBatcher",
]


class ProcessingBatcher(Generic[PackType], Configurable):
    r"""This defines the basis interface of the batcher used in
    :class:`~forte.processors.base.batch_processor.BaseBatchProcessor`. This
    Batcher only batches data sequentially. It receives new packs dynamically
    and cache the current packs so that the processors can pack prediction
    results into the data packs.
    """

    def __init__(self):
        super().__init__()
        self.current_batch: Dict = {}
        self.data_pack_pool: List[PackType] = []
        self.current_batch_sources: List[int] = []

        self._cross_pack: bool = True
        self.configs: Config = Config({}, {})

    def initialize(self, config: Union[Config, Dict]):
        r"""The implementation should initialize the batcher and setup the
        internal states of this batcher. This function will be called at the
        pipeline initialize stage.

        Returns:
            None
        """
        self.configs = self.make_configs(config)
        self._cross_pack = self.configs.cross_pack
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
            None
        """
        raise NotImplementedError

    def flush(
        self,
    ) -> Iterator[Tuple[List[PackType], List[Optional[Annotation]], Dict]]:
        r"""Flush the remaining data.

        Returns:
            A triplet contains datapack, context instance and batched data.

            .. note::

                For backward compatibility issues, this function
                return list of None contexts.
        """
        if self.current_batch:
            yield (
                self.data_pack_pool,
                [None] * len(self.data_pack_pool),
                self.current_batch,
            )
            self.current_batch = {}
            self.current_batch_sources = []
            self.data_pack_pool = []

    def get_batch(
        self, input_pack: PackType
    ) -> Iterator[Tuple[List[PackType], List[Optional[Annotation]], Dict]]:
        r"""By feeding data pack to this function, formatted features will
        be yielded based on the batching logic. Each element in the iterator is
        a triplet of datapack, context instance and batched data.

        Args:
            input_pack: The input data pack to get features from.

        Returns:
            An iterator of A tuple contains datapack, context instance and
            batch data.

            .. note::

                For backward compatibility issues, this function
                return a list of `None` as contexts.
        """
        batch_count = 0

        # cache the new pack and generate batches
        for (data_batch, instance_num) in self._get_data_batch(input_pack):
            self.current_batch = merge_batches([self.current_batch, data_batch])
            self.current_batch_sources.append(instance_num)
            self.data_pack_pool.extend([input_pack] * instance_num)

            # Yield a batch on two conditions.
            # 1. If we do not want to have batches from different pack, we
            # should yield since this pack is exhausted.
            # 2. We should also yield when the batcher condition is met:
            # i.e. ``_should_yield()`` is True.
            if not self._cross_pack or self._should_yield():
                batch_count += 1
                yield (
                    self.data_pack_pool,
                    [None] * len(self.data_pack_pool),
                    self.current_batch,
                )
                self.current_batch = {}
                self.current_batch_sources = []
                self.data_pack_pool = []

    def _get_data_batch(
        self,
        data_pack: PackType,
    ) -> Iterable[Tuple[Dict, int]]:
        r"""The abstract function that a batcher need to implement, to collect
        data from input data packs. It should yield data in the format of a
        tuple that contains the actual data points and the number of data
        points.

        These data points will be collected and organized in batches
        by the batcher, and can be obtained from the `get_batch` method.

        Args:
            data_pack: The data pack to retrieve data from.

        Returns:
            An iterator of tuples ``(batch, cnt)``, ``batch`` is a dict
            containing the required annotations and context, and ``cnt`` is
            the number of instances in the batch.
        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        Define the basic configuration of a batcher. Implementation of the
        batcher can extend this function to include more configurable
        parameters but need to keep the existing ones defined in this base
        class.

        Here, the available parameters are:

          - `use_coverage_index`: A boolean value indicates whether the
            batcher will try to build the coverage index based on the data
            request. Default is True.

          - `cross_pack`: A boolean value indicates whether the batcher can
            go across the boundary of data packs when there is no enough data
            to fill the batch.


        Returns:
            The default configuration.

        """
        return {
            "use_coverage_index": True,
            "cross_pack": True,
        }


class FixedSizeDataPackBatcher(ProcessingBatcher[DataPack]):
    def __init__(self):
        super().__init__()
        self.batch_is_full = False

    def initialize(self, config: Config):
        super().initialize(config)
        self.batch_is_full = False

    def _should_yield(self) -> bool:
        return self.batch_is_full

    @abstractmethod
    def _get_instance(self, data_pack: DataPack) -> Iterator[Dict[str, Any]]:
        """
        Get instance from the data pack. By default, this function will use
        the `requests` in configuration to get data. One can implement this
        function to extract data instance.

        Args:
            data_pack: The data pack to extract data from.

        Returns:
            None
        """
        raise NotImplementedError

    def _get_data_batch(
        self,
        data_pack: DataPack,
    ) -> Iterable[Tuple[Dict, int]]:
        r"""Get batches from a dataset  with ``batch_size``, It will yield data
        in the format of a tuple that contains the actual data points and the
        number of data points.

        The data points are generated by querying the data pack using the
        `context_type` and `requests` configuration via calling the
        :meth:`~forte.data.DataPack.get_data` method. Here, Each data point is
        in the same format returned by the `get_data` method, and the meaning
        of `context_type` and `requests` are exactly the same as the `get_data`
        method.

        Args:
            data_pack: The data pack to retrieve data from.

        Returns:
            An iterator of tuples ``(batch, cnt)``, ``batch`` is a dict
            containing the required entries and context, and ``cnt`` is
            the number of instances in the batch.
        """
        instances: List[Dict] = []
        current_size = sum(self.current_batch_sources)

        for data in self._get_instance(data_pack):
            instances.append(data)
            if len(instances) == self.configs.batch_size - current_size:
                batch = batch_instances(instances)
                self.batch_is_full = True
                yield batch, len(instances)
                instances = []
                self.batch_is_full = False

        # Flush the remaining data.
        if len(instances) > 0:
            batch = batch_instances(instances)
            yield batch, len(instances)

    @classmethod
    def default_configs(cls) -> Dict:
        """
        The configuration of a batcher.

        Here:

            - batch_size: the batch size, default is 10.


        Returns:
            The default configuration structure and default value.
        """
        return {
            "batch_size": 10,
        }


class FixedSizeRequestDataPackBatcher(FixedSizeDataPackBatcher):
    def initialize(self, config: Config):
        super().initialize(config)
        if self.configs.context_type is None:
            raise ProcessorConfigError(
                f"The 'context_type' config of {self.__class__.__name__} "
                f"cannot be None."
            )

    def _get_instance(self, data_pack: DataPack) -> Iterator[Dict[str, Any]]:
        """
        Get instance from the data pack. By default, this function will use
        the `requests` in configuration to get data. One can implement this
        function to extract data instance.

        Args:
            data_pack: The data pack to extract data from.

        Returns:
            None
        """
        yield from data_pack.get_data(
            self.configs.context_type, self.configs.requests.todict()
        )

    @classmethod
    def default_configs(cls) -> Dict:
        """
        The configuration of a batcher.

        Here:

            - context_type (str): The fully qualified name of an `Annotation`
              type, which will be used as the context to retrieve data from.
              For
              example, if a `ft.onto.Sentence` type is provided, then it will
              extract data within each sentence.
            - requests: The request detail. See
              :meth:`~forte.data.data_pack.DataPack.get_data` on what a request
              looks like.


        Returns:
            The default configuration structure and default value.
        """
        return {
            "context_type": None,
            "requests": {},
            "@no_typecheck": "requests",
        }


class FixedSizeMultiPackProcessingBatcher(ProcessingBatcher[MultiPack]):
    r"""A Batcher used in :class:`~forte.processors.base.batch_processor.MultiPackBatchProcessor`.

    .. note::

        this implementation is not finished.

    The Batcher calls the ProcessingBatcher inherently on each specified
    data pack in the MultiPack.

    It's flexible to query MultiPack so we delegate the task to the subclasses
    such as:

        - query all packs with the same ``context`` and ``input_info``.
        - query different packs with different ``context`` and
          ``input_info``.

    Since the batcher will save the data_pack_pool on the fly, it's not trivial
    to do batching and slicing multiple data packs in the same time
    """

    def __init__(self):
        super().__init__()
        self.batch_is_full = False
        self.input_pack_name: str = ""
        self.batch_size = -1

    def initialize(self, config: Config):
        super().initialize(config)
        self.batch_is_full = False

    def _should_yield(self) -> bool:
        return self.batch_is_full

    def _get_data_batch(
        self, multi_pack: MultiPack
    ) -> Iterable[Tuple[Dict, int]]:
        # TODO: Principled way of get data from multi pack?
        raise NotImplementedError

    @classmethod
    def default_configs(cls) -> Dict:
        return {"batch_size": 10}
