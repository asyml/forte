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
import itertools
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Type, Any

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data import slice_batch
from forte.data.base_pack import PackType
from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcher
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Annotation
from forte.data.types import DataRequest
from forte.process_manager import ProcessJobStatus
from forte.processors.base.base_processor import BaseProcessor

__all__ = [
    "BaseBatchProcessor",
    "BatchProcessor",
    "MultiPackBatchProcessor",
    "FixedSizeBatchProcessor",
    "FixedSizeMultiPackBatchProcessor"
]


class BaseBatchProcessor(BaseProcessor[PackType], ABC):
    r"""The base class of processors that process data in batch. This processor
    enables easy data batching via analyze the context and data objects. The
    context defines the scope of analysis of a particular task.

    For example, in
    dependency parsing, the context is normally a sentence, in entity
    coreference, the context is normally a document. The processor will create
    data batches relative to the context.

    Key fields in this processor:
        - context_type (Annotation): define the context (scope) to process.
        - input_info: A data request. Based on this input_info. If
          `use_coverage_index` is set to true, the processor will build the
          index based on the input information to speed up the entry
          searching time.
        - batcher: The processing batcher used for this processor.The batcher
          will also keep track of the relation between the pack and the batch
          data.
        - use_coverage_index: If true, the index will be built based on the
          input_info.
    """

    def __init__(self):
        super().__init__()
        self.context_type: Type[Annotation] = self._define_context()
        self.input_info: DataRequest = self._define_input_info()
        self.batcher: ProcessingBatcher = self.define_batcher()
        self.use_coverage_index = False

    def initialize(self, resources: Resources, configs: Optional[Config]):
        super().initialize(resources, configs)

        assert configs is not None
        try:
            self.batcher.initialize(configs.batcher)
        except AttributeError as e:
            raise ProcessorConfigError(
                "Error in handling batcher config, please provide the "
                "check the config to see if you have the key 'batcher'."
            ) from e

    @staticmethod
    @abstractmethod
    def _define_context() -> Type[Annotation]:
        r"""User should define the context type for batch processors here. The
        context must be of type :class:`Annotation`, the processor will create
        data batches with in the span of each annotations. For example, if the
        context type is ``Sentence``, and the task is POS tagging, then each
        batch will contain the POS tags for all words in the sentence.

        The "context" parameter here has the same meaning as the
        :meth:`get_data()` function in class :class:`DataPack`.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _define_input_info() -> DataRequest:
        r"""User should define the :attr:`input_info` for the batch processors
        here. The input info will be used to get batched data for this
        processor.

        The request here has the same meaning as the
        :meth:`get_data()` function in class :class:`DataPack`.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def define_batcher() -> ProcessingBatcher:
        r"""Define a specific batcher for this processor.
        Single pack :class:`BatchProcessor` initialize the batcher to be a
        :class:`~forte.data.batchers.ProcessingBatcher`.
        And :class:`MultiPackBatchProcessor` initialize the batcher to be a
        :class:`~forte.data.batchers.MultiPackProcessingBatcher`.
        """
        raise NotImplementedError

    def _process(self, input_pack: PackType):
        r"""In batch processors, all data are processed in batches. So this
        function is implemented to convert the input datapacks into batches
        according to the Batcher. Users do not need to implement this function
        but should instead implement ``predict``, which computes results from
        batches, and ``pack_all``, which convert the batch results back to
        datapacks.

        Args:
            input_pack: The next input pack to be fed in.
        """
        if self.use_coverage_index:
            self._prepare_coverage_index(input_pack)

        for packs, _, batch in self.batcher.get_batch(
                input_pack, self.context_type, self.input_info):
            pred = self.predict(batch)
            self.pack_all(packs, pred)

        # update the status of the jobs. The jobs which were removed from
        # data_pack_pool will have status "PROCESSED" else they are "QUEUED"
        q_index = self._process_manager.current_queue_index
        u_index = self._process_manager.unprocessed_queue_indices[q_index]
        data_pool_length = len(self.batcher.data_pack_pool)
        current_queue = self._process_manager.current_queue

        for i, job_i in enumerate(
                itertools.islice(current_queue, 0, u_index + 1)):
            if i <= u_index - data_pool_length:
                job_i.set_status(ProcessJobStatus.PROCESSED)
            else:
                job_i.set_status(ProcessJobStatus.QUEUED)

    def flush(self):
        for packs, _, batch in self.batcher.flush():
            pred = self.predict(batch)
            self.pack_all(packs, pred)

        current_queue = self._process_manager.current_queue

        for job in current_queue:
            job.set_status(ProcessJobStatus.PROCESSED)

    @abstractmethod
    def predict(self, data_batch: Dict) -> Dict:
        r"""The function that task processors should implement. Make
        predictions for the input ``data_batch``.

        Args:
              data_batch (dict): A batch of instances in our ``dict`` format.

        Returns:
              The prediction results in dict datasets.
        """
        pass

    def pack_all(self, packs: List[PackType], output_dict: Dict):
        r"""Pack the prediction results ``output_dict`` back to the
        corresponding packs.
        """
        data_pack_pool = []
        current_batch_sources = []
        prev_pack = None
        for pack_i in packs:
            if pack_i != prev_pack:
                current_batch_sources.append(1)
                prev_pack = pack_i
                data_pack_pool.append(pack_i)
            else:
                current_batch_sources[-1] += 1

        start = 0
        for i, pack_i in enumerate(data_pack_pool):
            output_dict_i = slice_batch(output_dict, start,
                                current_batch_sources[i])
            self.pack(pack_i, output_dict_i)
            start += current_batch_sources[i]
            pack_i.add_all_remaining_entries()

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        super_config = super().default_configs()

        super_config['batcher'] = cls.define_batcher().default_configs()

        return super_config

    @abstractmethod
    def pack(self, pack: PackType, inputs) -> None:
        r"""The function that task processors should implement.

        Add corresponding fields to ``pack``. Custom function of how
        to add the value back.

        Args:
            pack (PackType): The pack to add entries or fields to.
            inputs: The prediction results returned by :meth:`predict`. You
                need to add entries or fields corresponding to this prediction
                results to ``pack``.
        """
        raise NotImplementedError

    @abstractmethod
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


class BatchProcessor(BaseBatchProcessor[DataPack], ABC):
    r"""The batch processors that process :class:`DataPack`.
    """

    def _prepare_coverage_index(self, input_pack: DataPack):
        for entry_type in self.input_info.keys():
            input_pack.build_coverage_for(self.context_type, entry_type)


class FixedSizeBatchProcessor(BatchProcessor, ABC):
    @staticmethod
    def define_batcher() -> ProcessingBatcher:
        return FixedSizeDataPackBatcher()


class MultiPackBatchProcessor(BaseBatchProcessor[MultiPack], ABC):
    r"""This just defines the generic type to :class:`MultiPack`.
    The implemented batch processors will process :class:`MultiPack`.
    """

    def __init__(self):
        super().__init__()
        self.input_pack_name = None

    # TODO multi pack batcher need to be further studied.
    def _prepare_coverage_index(self, input_pack: MultiPack):
        for entry_type in self.input_info.keys():
            input_pack.packs[self.input_pack_name].build_coverage_for(
                self.context_type, entry_type)


class FixedSizeMultiPackBatchProcessor(MultiPackBatchProcessor, ABC):
    @staticmethod
    def define_batcher() -> ProcessingBatcher:
        return FixedSizeDataPackBatcher()
