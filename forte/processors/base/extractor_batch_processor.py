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
"""
The processors that process data in batch.
"""
import itertools
from abc import abstractmethod, ABC
from typing import Dict, Optional, Any

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.types import DATA_INPUT
from forte.data.base_pack import PackType
from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcher
from forte.data.data_pack import DataPack
from forte.process_manager import ProcessJobStatus
from forte.processors.base.base_processor import BaseProcessor

__all__ = [
    "BaseBatchProcessor",
    "BatchProcessor",
    "FixedSizeBatchProcessor",
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
        self.batcher: ProcessingBatcher = self.define_batcher()
        self.use_coverage_index = False

    def initialize(self, resources: Resources, configs: Optional[Config]):
        super().initialize(resources, configs)
        self.configs = configs.processor
        self.batcher.initialize(configs.batcher)

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

        for batch in self.batcher.get_batch(input_pack):
            packs, instances, features = batch
            predictions = self.predict(features)
            for tag, preds in predictions.items():
                for pred, pack, instance in zip(preds, packs, instances):
                    self.configs.feature_scheme[tag]["extractor"].add_to_pack(
                        pack, instance, pred)
                    pack.add_all_remaining_entries()

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
        for batch in self.batcher.flush():
            packs, instances, features = batch
            predictions = self.predict(features)
            for tag, preds in predictions.items():
                for pred, pack, instance in zip(preds, packs, instances):
                    self.configs.feature_scheme[tag]["extractor"].add_to_pack(
                        pack, instance, pred)
                    pack.add_all_remaining_entries()

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

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        super_config = super().default_configs()
        super_config['processor'] = {
            "scope": None,
            "feature_scheme": {}
        }
        super_config['batcher'] = cls.define_batcher().default_configs()

        return super_config

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
        for _, scheme in self.configs.feature_scheme:
            input_pack.build_coverage_for(self.configs.scope,
                                scheme["extractor"].entry_type)


class FixedSizeBatchProcessor(BatchProcessor, ABC):
    @staticmethod
    def define_batcher() -> ProcessingBatcher:
        return FixedSizeDataPackBatcher()
