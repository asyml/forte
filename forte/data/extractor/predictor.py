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


from typing import Dict, List, Callable
import itertools
from torch.nn import Module
from forte.data.types import DATA_INPUT
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.processors.base.base_processor import BaseProcessor
from forte.process_manager import ProcessJobStatus

class Batcher(object):
    '''This class will create a pool of pack, instance and feature. When
    the pool is filled with a batch size of elements. It will generate them
    in a batch. The batched data will be pass to the prediction function in the
    Predictor. Note that the extract, convert process is done during this batching
    process.
    '''
    def __init__(self, batch_size: int, feature_resource: Dict):
        '''Feature_resource is prodcued from train pipeline. An example looks like
        {
        'scope': ft.onto.base_ontology.Sentence,
        'schemes': {
            'text_tag': {
                'extractor': <forte.data.extractor.extractor.TextExtractor>,
                'converter': <forte.data.extractor.converter.Converter>,
                'type': DATA_INPUT
            },

            'char_tag': {
                'extractor': <forte.data.extractor.extractor.CharExtractor>,
                'converter': <forte.data.extractor.converter.Converter>,
                'type': DATA_INPUT
            },

            'ner_tag': {
                'extractor': <forte.data.extractor.extractor.AnnotationSeqExtractor>,
                'converter': <forte.data.extractor.converter.Converter>,
                'type': DATA_OUTPUT,
                'unpadder': <forte.data.extractor.converter.SameLengthUnpadder>}
            }
        }
        '''
        self.batch_size = batch_size
        self.feature_resource = feature_resource
        self.pack_pools = []
        self.instance_pools = []
        self.features_pools = []

    def current_pack_len(self) -> int:
        return len(set(self.pack_pools))

    def convert(self, feature_collections: List) -> Dict:
        example_collection = {}
        for features_collection in feature_collections:
            for tag, feature in features_collection.items():
                if tag not in example_collection:
                    example_collection[tag] = []
                example_collection[tag].append(feature)

        tensor_collection = {}
        for tag, features in example_collection.items():
            converter = self.feature_resource['schemes'][tag]["converter"]
            tensor, mask = converter.convert(features)
            if tag not in tensor_collection:
                tensor_collection[tag] = {}
            tensor_collection[tag]["tensor"] = tensor
            tensor_collection[tag]["mask"] = mask
        return tensor_collection

    def yield_batch(self, pack: DataPack) -> Dict:
        for instance in pack.get(self.feature_resource['scope']):
            feature_collection = {}
            for tag, scheme in self.feature_resource['schemes'].items():
                if scheme['type'] == DATA_INPUT:
                    extractor = scheme["extractor"]
                    feature = extractor.extract(pack, instance)
                    feature_collection[tag] = feature

            self.pack_pools.append(pack)
            self.instance_pools.append(instance)
            self.features_pools.append(feature_collection)

            if len(self.features_pools) == self.batch_size:
                yield self.convert(self.features_pools[:self.batch_size]), \
                    self.pack_pools[:self.batch_size], \
                    self.instance_pools[:self.batch_size]

                self.features_pools.clear()
                self.pack_pools.clear()
                self.instance_pools.clear()

    def flush_batch(self) -> Dict:
        if len(self.features_pools) > 0:
            yield self.convert(self.features_pools), self.pack_pools, \
                self.instance_pools
            self.features_pools.clear()
            self.pack_pools.clear()
            self.instance_pools.clear()


class Predictor(BaseProcessor):
    '''This class will predict using the passed in model and add output
    back to the datapack.
    '''
    def __init__(self, batch_size: int, pretrain_model: Module,
            predict_forward_fn: Callable, feature_resource: Dict):
        super().__init__()
        self.feature_resource = feature_resource
        self.batcher = Batcher(batch_size = batch_size,
                                feature_resource = feature_resource)
        self.pretrain_model = pretrain_model
        self.predict_forward_fn = predict_forward_fn

    def unpad(self, predictions: Dict, packs: List[DataPack],
                    instances: List[Annotation]):
        new_predictions = {}
        for tag, preds in predictions.items():
            new_preds = []
            for pred, pack, instance in zip(preds, packs, instances):
                new_preds.append(
                    self.feature_resource['schemes'][tag]['unpadder'].unpad(
                        pred, pack, instance))
            new_predictions[tag] = new_preds
        return new_predictions

    def add_to_pack(self, predictions: Dict, packs: List[DataPack],
                    instances: List[Annotation]):
        for tag, preds in predictions.items():
            for pred, pack, instance in zip(preds, packs, instances):
                self.feature_resource['schemes'][tag]['extractor'].add_to_pack(
                            pack, instance, pred)
                pack.add_all_remaining_entries()

    def _process(self, input_pack: DataPack):
        for tensor_collection, packs, instances in \
                                self.batcher.yield_batch(input_pack):
            predictions = self.predict_forward_fn(self.pretrain_model, tensor_collection)
            predictions = self.unpad(predictions, packs, instances)
            self.add_to_pack(predictions, packs, instances)

        # update the status of the jobs. The jobs which were removed from
        # data_pack_pool will have status "PROCESSED" else they are "QUEUED"
        q_index = self._process_manager.current_queue_index
        u_index = self._process_manager.unprocessed_queue_indices[q_index]
        data_pool_length = self.batcher.current_pack_len()
        current_queue = self._process_manager.current_queue

        for i, job_i in enumerate(
                itertools.islice(current_queue, 0, u_index + 1)):
            if i <= u_index - data_pool_length:
                job_i.set_status(ProcessJobStatus.PROCESSED)
            else:
                job_i.set_status(ProcessJobStatus.QUEUED)

    def flush(self):
        for tensor_collection, packs, instances in self.batcher.flush_batch():
            predictions = self.predict_forward_fn(self.pretrain_model, tensor_collection)
            predictions = self.unpad(predictions, packs, instances)
            self.add_to_pack(predictions, packs, instances)

        current_queue = self._process_manager.current_queue
        for job in current_queue:
            job.set_status(ProcessJobStatus.PROCESSED)
