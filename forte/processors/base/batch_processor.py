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
The processors that process data in batch.
"""
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Any

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data import slice_batch, BaseExtractor
from forte.data.base_pack import PackType
from forte.data.batchers import (
    ProcessingBatcher,
    FixedSizeDataPackBatcher,
    FixedSizeDataPackBatcherWithExtractor,
)
from forte.data.converter import Converter
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Annotation
from forte.processors.base.base_processor import BaseProcessor
from forte.utils import extractor_utils

__all__ = [
    "BaseBatchProcessor",
    "PackingBatchProcessor",
    "Predictor",
    "MultiPackBatchProcessor",
    "FixedSizeBatchPackingProcessor",
]

from forte.utils.extractor_utils import (
    parse_feature_extractors,
)


class BaseBatchProcessor(BaseProcessor[PackType], ABC):
    r"""The base class of processors that process data in batch. This processor
    enables easy data batching via analyze the context and data objects. The
    context defines the scope of analysis of a particular task.

    For example, in dependency parsing, the context is normally a sentence,
    in entity coreference, the context is normally a document. The processor
    will create data batches relative to the context.

    Key fields in this processor:

        - batcher: The processing batcher used for this processor. The batcher
          will also keep track of the relation between the pack and the batch
          data.

        - use_coverage_index: If true, the index will be built based on the
          requests.
    """

    def __init__(self):
        super().__init__()
        self._batcher: Optional[ProcessingBatcher] = None
        self.use_coverage_index = False

    def initialize(self, resources: Resources, configs: Optional[Config]):
        super().initialize(resources, configs)

        assert configs is not None
        try:
            self.batcher.initialize(configs.batcher)
        except AttributeError as e:
            raise ProcessorConfigError(
                "Error in handling batcher config, please check the "
                "config of the batcher to see they are correct."
            ) from e

    @property
    def batcher(self) -> ProcessingBatcher:
        if self._batcher is None:
            self._batcher = self.define_batcher()
        return self._batcher

    def _process(self, input_pack: PackType):
        r"""In batch processors, all data are processed in batches. This
        function is already implemented to convert data in to batches. Users do
        not need to implement this function, but should instead implement
        ``_process_batch``.

        but should instead implement ``predict``, which computes results from
        batches, and ``pack_all``, which convert the batch results back to
        datapacks.

        Args:
            input_pack: The next input pack to be fed in.
        """
        if self.use_coverage_index:
            self._prepare_coverage_index(input_pack)

        for packs, instances, batch in self.batcher.get_batch(input_pack):
            self._process_batch(packs, instances, batch)

    def flush(self):
        for packs, instances, batch in self.batcher.flush():
            self._process_batch(packs, instances, batch)

    @abstractmethod
    def _process_batch(
        self,
        packs: List[PackType],
        contexts: List[Optional[Annotation]],
        batched_data: Dict,
    ):
        """
        Users can implement this function to process the extracted batch
        data. This is suitable to be implemented if one do not need to add
        information back to the data pack. Otherwise, It is advised to implement
        :class:`~forte/processors/base.batch_processor.PackingBatchProcessor`
        instead of this one since PackingBatchProcessor provides helper
        functions to re-align the output with the data pack.

        Args:
            packs: The list of data packs corresponding to the batch.
            contexts: The list of context corresponding to the batch. It
                could contain `None` values, which means the the whole data
                pack will be used as contexts.
            batched_data: The data batch to be process.

        Returns:

        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        r"""Defines the default configs for batching processor."""
        super_config = super().default_configs()
        super_config["batcher"] = cls.define_batcher().default_configs()
        return super_config

    def _prepare_coverage_index(self, input_pack: PackType):
        """
        Build the coverage index for ``input_pack``. After building, querying
          data in this pack will become more efficient.

        The index will be built based on the `input_info` field.

        Args:
            input_pack: The pack to be built.

        Returns:

        """
        pass

    @classmethod
    @abstractmethod
    def define_batcher(cls) -> ProcessingBatcher:
        r"""Define a specific batcher for this processor.
        Single pack :class:`BatchProcessor` initialize the batcher to be a
        :class:`~forte.data.batchers.ProcessingBatcher`.
        And :class:`MultiPackBatchProcessor` initialize the batcher to be a
        :class:`~forte.data.batchers.MultiPackProcessingBatcher`.
        """
        raise NotImplementedError


class PackingBatchProcessor(BaseBatchProcessor[PackType], ABC):
    """
    This class extend the BaseBatchProcessor class and provide additional
    utilities to align and pack the extracted results back to the data pack.

    To implement this processor, one need to implement:
    1. The `predict` function that make predictions for each input data batch.
    2. The `pack` function that add the prediction value back to the dat pack.

    Users that implement the processor only have to concern about a single
    batch, the alignment between the data batch and the data pack will be
    maintained by the system.
    """

    def _process_batch(
        self,
        packs: List[PackType],
        contexts: List[Optional[Annotation]],
        batched_data: Dict,
    ):
        """
        Implement the function into a prediction and pack step, users don't
        need to implement this function and should implement `predict` and
        `pack` instead.

        Args:
            packs (List[PackType]):  List of data packs each batch comes from.
            batched_data (Dict): The batched data.
            contexts (Optional[List[Annotation]]): List of the data contexts
                where the data comes from.

        Returns:

        """
        pred = self.predict(batched_data)
        self.pack_all(packs, contexts, pred)

    def predict(self, data_batch: Dict) -> Dict[str, List[Any]]:
        r"""The function that task processors should implement. Make
        predictions for the input ``data_batch``.

        Args:
              data_batch (dict): A batch of instances in our ``dict`` format.

        Returns:
              The prediction results in dictionary form.
        """
        raise NotImplementedError

    def pack(
        self,
        pack: PackType,
        predict_results: Dict[str, List[Any]],
        context: Optional[Annotation] = None,
    ):
        r"""The function that task processors should implement. It is the
        custom function on how to add the predicted output back to the data
        pack.

        Args:
            pack (PackType): The pack to add entries or fields to.
            predict_results (Dict): The prediction results returned by
                :meth:`~forte.processors.base.batch_processor
                .BaseBatchProcessor.predict`.
                This processor will add these results to the provided `pack`
                as entry and attributes.
            context (Optional[Annotation]): The context entry that the
                prediction is performed, and the pack operation should
                be performed related to this range annotation. If None,
                then we consider the whole data pack is used as the context.
        """
        raise NotImplementedError

    def pack_all(
        self,
        packs: List[PackType],
        contexts: List[Optional[Annotation]],
        output_dict: Dict[str, List[Any]],
    ):
        r"""
        Pack the prediction results contained in the `output_dict` back to the
        corresponding packs.

        Args:
            packs: The list of data packs corresponding to the output batches.
            contexts: The list of contexts corresponding to the output batches.
            output_dict: Stores the output in a specific format. The keys
                are string names that specify data. The value is a list of
                data in the shape of (batch_size, Any). There might be
                additional structures inside `Any` as specific
                implementation choices.
        """
        # Group the same pack and context into the same segments.

        # The list of (pack, context) tuple.
        pack_context_pool = []
        # Store the segments of the pack context, the len of this list should
        # be the number of segments, and the value indicates the length of
        # the segment. These will be used to slice the data batch.
        segment_lengths = []

        # Note that this will work if the elements in `contexts` are all
        # None, which means they are all the same, and will have the correct
        # behavior.
        prev_pack_context = None
        for pack_context_i in zip(packs, contexts):
            if pack_context_i != prev_pack_context:
                segment_lengths.append(1)
                prev_pack_context = pack_context_i
                pack_context_pool.append(pack_context_i)
            else:
                segment_lengths[-1] += 1

        start = 0

        for i, (pack_i, context) in enumerate(pack_context_pool):
            # The slice should correspond to the portion of the data batch
            # that should be assigned to these pack and context.
            output_dict_i = slice_batch(output_dict, start, segment_lengths[i])
            self.pack(pack_i, output_dict_i, context)
            start += segment_lengths[i]
            pack_i.add_all_remaining_entries()


class FixedSizeBatchPackingProcessor(PackingBatchProcessor[DataPack], ABC):
    """
    A processor that implements the packing batch processor, using a fixed
    size batcher :class:`~forte.data.batchers.FixedSizeDataPackBatcher`
    """

    @classmethod
    def define_batcher(cls) -> ProcessingBatcher:
        return FixedSizeDataPackBatcher()


class Predictor(PackingBatchProcessor[PackType]):
    r"""
    `Predictor` is a special type of batch processor that uses
    :class:`~forte.data.BaseExtractor` to collect features from data packs, and
    also uses Extractors to write the prediction back.

    `Predictor` implements the `PackingBatchProcessor` class, and implements
    the `predict` and `pack` function using the extractors.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.do_eval = False
        self._request: Dict = {}
        self._request_ready: bool = False

        self._batcher: Optional[FixedSizeDataPackBatcherWithExtractor] = None

    def add_extractor(
        self,
        name: str,
        extractor: BaseExtractor,
        is_input: bool,
        converter: Optional[Converter] = None,
    ):
        """
        Extractors can be added to the preprocessor directly via this
        method.

        Args:
            name: The name/identifier of this extractor, the name should be
              different between different extractors.
            extractor: The extractor instance to be added.
            is_input: Whether this extractor will be used as input or output.
            converter:  The converter instance to be applied after running
              the extractor.

        Returns:

        """
        extractor_utils.add_extractor(
            self._request, name, extractor, is_input, converter
        )

    def set_feature_requests(self, request: Dict):
        self._request = request
        self._request_ready = True

    def deactivate_request(self):
        self._request_ready = False

    @classmethod
    def define_batcher(cls) -> ProcessingBatcher:
        return FixedSizeDataPackBatcherWithExtractor()

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        super_config = super().default_configs()
        super_config.update(
            {
                "feature_scheme": None,
                "context_type": None,
                "batcher": cls.define_batcher().default_configs(),
                "do_eval": False,
            }
        )
        return super_config

    def initialize(self, resources: Resources, configs: Config):
        # Populate the _request. The self._request_ready help avoid parsing
        # the feature scheme multiple times during `initialize`.
        if not self._request_ready:
            for key, value in configs.items():
                if key == "feature_scheme":
                    self._request["schemes"] = parse_feature_extractors(
                        configs.feature_scheme
                    )
                else:
                    self._request[key] = value
            self._request_ready = True

        batcher_config = configs.batcher
        # Assign context type from here to make sure batcher is using the
        # same context type as predictor.
        batcher_context = configs["batcher"].get("context_type", None)
        if (
            batcher_context is None
            or batcher_context == self._request["context_type"]
        ):
            batcher_config.context_type = self._request["context_type"]
        else:
            raise ProcessorConfigError(
                "The 'context_type' configuration value should be the same "
                "for the processor and the batcher, now for the processor the "
                f"value is {self._request['context_type']} and for the "
                f"batcher the value is {batcher_context}. It is also fine if "
                f"this value for batch config is left empty."
            )
        self.do_eval = configs.do_eval

        # This needs to be called later since batcher config needs to be loaded.
        super().initialize(resources, configs)
        for tag, scheme in self._request["schemes"].items():
            # Add input feature to the batcher.
            if scheme["type"] == extractor_utils.DATA_INPUT:
                self.batcher.add_feature_scheme(tag, scheme)  # type: ignore

    def load(self, model):
        self.model = model

    def pack(
        self,
        pack: PackType,
        predict_results: Dict,
        context: Optional[Annotation] = None,
    ):
        for tag, batched_predictions in predict_results.items():
            # preds contains batched results.
            if self.do_eval:
                self.__extractor(tag).pre_evaluation_action(pack, context)
            for prediction in batched_predictions:
                self.__extractor(tag).add_to_pack(pack, prediction, context)
        pack.add_all_remaining_entries()

    def __extractor(self, tag_name: str):
        return self._request["schemes"][tag_name]["extractor"]

    def predict(self, data_batch: Dict) -> Dict:
        r"""The function that task processors should implement. Make
        predictions for the input ``data_batch``.

        Args:
              data_batch (dict): A batch of instances in our ``dict`` format.

        Returns:
              The prediction results in dict datasets.
        """
        raise NotImplementedError()


class MultiPackBatchProcessor(BaseBatchProcessor[MultiPack], ABC):
    r"""This class defines the base batch processor for `MultiPack`s."""

    def __init__(self):
        super().__init__()
        self.input_pack_name = None
        # TODO multi pack batcher need to be further implemented.
