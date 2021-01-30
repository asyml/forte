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


from typing import List, Dict, Optional, Type, Any
from forte.common import Resources
from forte.common.configuration import Config
from forte.data.ontology.top import Annotation
from forte.data.ontology.core import Sentence
from forte.processors.base.batch_processor import BaseBatchProcessor
from forte.data.TrainPreprocessor import DATA_INPUT
from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcherWithExtractor


class Predictor(BaseBatchProcessor):
    @staticmethod
    def _define_context() -> Type[Annotation]:
        r"""This function is just for the compatibility reason.
        And it is not used.
        """
        return Sentence

    @staticmethod
    def define_batcher() -> ProcessingBatcher:
        return FixedSizeDataPackBatcherWithExtractor()

    def initialize(self, resources: Resources, configs: Optional[Config]):
        new_config = {}

        processor_config = {}
        processor_config["scope"] = configs.scope
        processor_config["feature_scheme"] = configs.feature_scheme

        batcher_config = {}
        batcher_config["scope"] = configs.scope
        batcher_config["feature_scheme"] = {}
        for tag, scheme in configs.feature_scheme.items():
            if scheme["type"] == DATA_INPUT:
                batcher_config["feature_scheme"][tag] = scheme
        batcher_config["batch_size"] = configs.batch_size

        new_config["processor"] = processor_config
        new_config["batcher"] = batcher_config

        super().initialize(resources, configs)

        assert "model" in configs
        self.model = configs.model

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
