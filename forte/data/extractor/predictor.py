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

# pylint: disable=logging-fstring-interpolation
import logging
import os
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Type, Any, Union, Iterable, Generic, Iterator

import numpy as np
import torch
import itertools
from copy import copy

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.data.ontology.core import Entry
from forte.data.types import DataRequest
from forte.models.ner import utils
from forte.data.base_pack import PackType
from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.processors.base.base_processor import BaseProcessor
from ft.onto.base_ontology import Token, Sentence, EntityMention
from forte.data.extractor.extractor import TextExtractor, AnnotationSeqExtractor
from forte.data.extractor.converter import Converter
from forte.process_manager import ProcessJobStatus
from forte.data import slice_batch
from forte.data.data_utils_io import merge_batches, batch_instances


class Predictor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.pack_pools = []
        self.instance_pools = []
        self.features_pools = []

    def initialize(self, resources: Resources, configs: Optional[Config]):
        super().initialize(resources, configs)
        # TODO: load these params from outside
        self.batch_size = 11
        self.scope = Sentence

        config1 = {
            "scope": Sentence,
            "entry": Token,
            "vocab_use_unk": True
        }
        extractor1 = TextExtractor(config1)
        converter1 = Converter(need_pad=True)

        config2 = {
            "scope": Sentence,
            "entry": EntityMention,
            "attribute": "ner_type",
            "based_on": Token,
            "strategy": "BIO",
            "vocab_use_unk": True
        }
        extractor2 = AnnotationSeqExtractor(config2)
        extractor2.add_entry(("PER", "B"))
        converter2 = Converter(need_pad=True)

        self.feature_scheme = {"text_tag": {
                                    "extractor":  extractor1,
                                    "converter": converter1
                                },
                                "ner_tag": {
                                    "extractor":  extractor2,
                                    "converter": converter2
                                }
                            }

        class Model:
            def __call__(self, tensor):
                # Input shape: [Batch, length]
                # Output shape: [Batch]
                batch_size = len(tensor)
                length = len(tensor[0])
                return [[2 for _ in range(length)] for _ in range(batch_size)]
        self.model = Model()

    def convert(self, feature_collections):
        example_collection = {}
        for features_collection in feature_collections:
            for tag, feature in features_collection.items():
                if tag not in example_collection:
                    example_collection[tag] = []
                example_collection[tag].append(feature)

        tensor_collection = {}
        for tag, features in example_collection.items():
            converter = self.feature_scheme[tag]["converter"]
            tensor, mask = converter.convert(features)

            if tag not in tensor_collection:
                tensor_collection[tag] = {}

            tensor_collection[tag]["tensor"] = tensor
            tensor_collection[tag]["mask"] = mask
        return tensor_collection

    def yield_batch(self, pack):
        for instance in pack.get(self.scope):
            feature_collection = {}
            for tag, scheme in self.feature_scheme.items():
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

    def flush_batch(self):
        if len(self.features_pools) > 0:
            yield self.convert(self.features_pools), self.pack_pools, \
                self.instance_pools
            self.features_pools.clear()
            self.pack_pools.clear()
            self.instance_pools.clear()

    def _process(self, input_pack: PackType):
        # pack = input_pack
        # for instance in pack.get(self.scope):
        #     feature_collection = {}
        #     for tag, scheme in self.feature_scheme.items():
        #         extractor = scheme["extractor"]
        #         feature = extractor.extract(pack, instance)
        #         feature_collection[tag] = feature
            
        #     self.pack_pools.append(pack)
        #     self.instance_pools.append(instance)
        #     self.features_pools.append(feature_collection)

        # predictions = self.predict(self.convert(self.features_pools))
        # self.add_to_pack(predictions, self.pack_pools, self.instance_pools)

        for tensor_collection, packs, instances in self.yield_batch(input_pack):
            predictions = self.predict(tensor_collection)
            self.add_to_pack(predictions, tensor_collection, packs, instances)

        # update the status of the jobs. The jobs which were removed from
        # data_pack_pool will have status "PROCESSED" else they are "QUEUED"
        q_index = self._process_manager.current_queue_index
        u_index = self._process_manager.unprocessed_queue_indices[q_index]
        data_pool_length = len(set(self.pack_pools))
        current_queue = self._process_manager.current_queue

        for i, job_i in enumerate(
                itertools.islice(current_queue, 0, u_index + 1)):
            if i <= u_index - data_pool_length:
                job_i.set_status(ProcessJobStatus.PROCESSED)
            else:
                job_i.set_status(ProcessJobStatus.QUEUED)

    def flush(self):
        for tensor_collection, packs, instances in self.flush_batch():

            predictions = self.predict(tensor_collection)
            self.add_to_pack(predictions, tensor_collection, packs, instances)
            pack = packs[0]

        current_queue = self._process_manager.current_queue
        for job in current_queue:
            job.set_status(ProcessJobStatus.PROCESSED)

    def predict(self, tensor_collection):
        predictions = self.model(tensor_collection["text_tag"]["tensor"])
        return predictions


    def add_to_pack(self, predictions, tensor_collection, packs, instances):
        prev_pack = None
        for pred, mask, pack, instance in zip(predictions,
                                        tensor_collection["text_tag"]["mask"],
                                         packs, instances):
            self.feature_scheme["ner_tag"]["extractor"].add_to_pack(
                pack, instance, pred[:sum(mask)]
            )
            if prev_pack is not None and prev_pack != pack:
                prev_pack.add_all_remaining_entries()
            prev_pack = pack

        if prev_pack is not None:
            prev_pack.add_all_remaining_entries()

    def new_pack(self, pack_name: Optional[str] = None) -> DataPack:
        return DataPack(pack_name)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        super_config = super().default_configs()
        return super_config


    # def should_yield(self):
    #     return len(self.features_pools) >= self.batch_size

    # def remain_pack_num(self):
    #     return len(self.pack_pools)

    # def clean_pack_pools(self):
    #     if self.remain_pack_num() > 0:
    #         cut_id = 0
    #         while cut_id < len(self.pack_pools) and \
    #             self.pack_pools[cut_id][1] == 0:
    #             cut_id += 1
    #         self.pack_pools = self.pack_pools[cut_id:]

    # def create_batch(self, flush=False):
    #     if flush:
    #         size = len(self.features_pools)
    #     else:
    #         size = self.batch_size

    #     example_collection = {}
    #     for features_collection in self.features_pools[:size]:
    #         for tag, feature in features_collection.items():
    #             if tag not in example_collection:
    #                 example_collection[tag] = []
    #             example_collection[tag].append(feature)

    #     tensor_collection = {}
    #     for tag, features in example_collection.items():
    #         converter = self.feature_scheme[tag]["converter"]
    #         tensor, mask = converter.convert(features)

    #         if tag not in tensor_collection:
    #             tensor_collection[tag] = {}

    #         tensor_collection[tag]["tensor"] = tensor
    #         tensor_collection[tag]["mask"] = mask

    #     self.features_pools = self.features_pools[size:]

    #     pack_info = []
    #     pack_id = 0
    #     while size > 0:
    #         if self.pack_pools[pack_id][1] > 0 and \
    #             self.pack_pools[pack_id][1] <= size:
    #             pack_info.append([self.pack_pools[pack_id][0],
    #                                 self.pack_pools[pack_id][1]])
    #             size -= self.pack_pools[pack_id][1]
    #             self.pack_pools[pack_id][1] = 0
    #         elif self.pack_pools[pack_id][1] > 0 and \
    #             self.pack_pools[pack_id][1] > size:
    #             pack_info.append([self.pack_pools[pack_id][0],
    #                                 size])
    #             self.pack_pools[pack_id][1] -= size
    #             size = 0
    #         pack_id += 1

    #     if flush:
    #         self.clean_pack_pools()

    #     return tensor_collection, pack_info

    # def pack_all(self, output_dict: Dict):
    #     r"""Pack the prediction results ``output_dict`` back to the
    #     corresponding packs.
    #     """
    #     start = 0
    #     for i in range(len(self.batcher.data_pack_pool)):
    #         pack_i = self.batcher.data_pack_pool[i]
    #         output_dict_i = slice_batch(output_dict, start,
    #                                     self.batcher.current_batch_sources[i])
    #         self.pack(pack_i, output_dict_i)
    #         start += self.batcher.current_batch_sources[i]
    #         pack_i.add_all_remaining_entries()

    # def flush(self):
    #     print("=====flush is called" )
    #     for batch in self.batcher.flush():
    #         pred = self.predict(batch)
    #         self.pack_all(pred)
    #         self.update_batcher_pool(-1)

    #     current_queue = self._process_manager.current_queue

    #     for job in current_queue:
    #         job.set_status(ProcessJobStatus.PROCESSED)


    # def update_batcher_pool(self, end: Optional[int] = None):
    #     r"""Update the batcher pool in :attr:`data_pack_pool` from the
    #     beginning to ``end`` (``end`` is not included).

    #     Args:
    #         end (int): Will do finishing work for data packs in
    #             :attr:`data_pack_pool` from the beginning to ``end``
    #             (``end`` is not included). If `None`, will finish up all the
    #             packs in :attr:`data_pack_pool`.
    #     """
    #     # TODO: the purpose of this function is confusing, especially the -1
    #     #  argument value.
    #     if end is None:
    #         end = len(self.batcher.data_pack_pool)
    #     self.batcher.data_pack_pool = self.batcher.data_pack_pool[end:]
    #     self.batcher.current_batch_sources = \
    #         self.batcher.current_batch_sources[end:]

    # def prepare_coverage_index(self, input_pack: DataPack):
    #     for entry_type in self.input_info.keys():
    #         if input_pack.index.coverage_index(self.context_type,
    #                                            entry_type) is None:
    #             input_pack.index.build_coverage_index(
    #                 input_pack,
    #                 self.context_type,
    #                 entry_type
    #             )


# Implementation using BaseProcessor
# class Predictor(BaseProcessor):
#     """
#        An Named Entity Recognizer trained according to `Ma, Xuezhe, and Eduard
#        Hovy. "End-to-end sequence labeling via bi-directional lstm-cnns-crf."
#        <https://arxiv.org/abs/1603.01354>`_.

#        Note that to use :class:`CoNLLNERPredictor`, the :attr:`ontology` of
#        :class:`Pipeline` must be an ontology that include
#        ``ft.onto.base_ontology.Token`` and ``ft.onto.base_ontology.Sentence``.
#     """

#     def __init__(self):
#         super().__init__()
#         self.model = None
#         self.word_alphabet, self.char_alphabet, self.ner_alphabet = (
#             None, None, None)
#         self.resource = None

#     def initialize(self, resource: Resources, configs: Config):
#         self.resource = resource
#         self.configs = configs
#         print("configs", configs)
#         print("resource", resource.resources)
#         self.extractors = self.load_extractors(configs)
#         self.model = self.load_model(configs)

#     def load_extractors(self, configs):
#         config1 = {
#             "scope": Sentence,
#             "entry": EntityMention,
#             "attribute": "ner_type",
#             "based_on": Token,
#             "strategy": "BIO",
#             "vocab_use_unk": True
#         }
#         extractor1 = AnnotationSeqExtractor(config1)
#         extractor1.add_entry(("PER", "B"))
#         config0 = {
#             "scope": Sentence,
#             "entry": Token,
#             "vocab_use_unk": True
#         }
#         extractor0 = TextExtractor(config0)
#         return [extractor0, extractor1]

#     def load_model(self, configs):
#         class Model:
#             def __call__(self, tensor):
#                 # Input shape: [Batch, length]
#                 # Output shape: [Batch]
#                 batch_size = len(tensor)
#                 length = len(tensor[0])
#                 return [[2 for _ in range(length)] for _ in range(batch_size)]
#         return Model()


#     def new_pack(self, pack_name: Optional[str] = None) -> DataPack:
#         return DataPack(pack_name)

#     def _process(self, input_pack: PackType):
#         for instance in input_pack.get(Sentence):
#             feat0 = self.extractors[0].extract(input_pack, instance)
#             feat1 = self.extractors[1].extract(input_pack, instance)
#             input_tensor = [feat0.data] # add batch dim
#             output_tensor = self.model(input_tensor)[0] # extract batch dim
#             feat1.data = output_tensor
#             self.extractors[1].add_to_pack(input_pack, instance, feat1)
#         return input_pack

#     @classmethod
#     def default_configs(cls):
#         r"""Default config for NER Predictor"""

#         configs = super().default_configs()
#         # TODO: Batcher in NER need to be update to use the sytem one.

#         more_configs = {
#             "config_data": {
#                 "train_path": "",
#                 "val_path": "",
#                 "test_path": "",
#                 "num_epochs": 200,
#                 "batch_size_tokens": 512,
#                 "test_batch_size": 16,
#                 "max_char_length": 45,
#                 "num_char_pad": 2
#             },
#         }

#         configs.update(more_configs)
#         return configs



# class Batcher:
#     def __init__(self, batch_size):
#         self.batch_size = batch_size

#         # TODO: load these params from outside
#         self.get_data_args = {"context_type": Sentence,
#                                 "request": {Sentence: []},
#                                 "skip_k": 0}

#         config1 = {
#             "scope": Sentence,
#             "entry": Token,
#             "vocab_use_unk": True
#         }
#         extractor1 = TextExtractor(config1)
#         converter1 = Converter(need_pad=True)

#         config2 = {
#             "scope": Sentence,
#             "entry": EntityMention,
#             "attribute": "ner_type",
#             "based_on": Token,
#             "strategy": "BIO",
#             "vocab_use_unk": True
#         }
#         extractor2 = AnnotationSeqExtractor(config2)
#         extractor2.add_entry(("PER", "B"))
#         converter2 = Converter(need_pad=True)

#         self.feature_scheme = {"text_tag": {
#                                     "extractor":  extractor1,
#                                     "converter": Converter1
#                                 },
#                                 "ner_tag": {
#                                     "extractor":  extractor2,
#                                     "converter": converter2
#                                 }
#                             }

#         self.pack_pools = []
#         self.features_pools = []

#     def initialize(self):
#         self.pack_pools.clear()
#         self.features_pools.clear()

#     def flush(self):
#         if len(self.features_pools) > 0:
#             return self.create_batch(flush=True)
#         return None

#     def get_batch(self, pack):
#         self.pack_pools.append([pack, 0])

#         for instance in pack.get_data(**self.get_data_args):
#             feature_collection = {}
#             for tag, scheme in self.feature_scheme.items():
#                 extractor = scheme["extractor"]
#                 feature = extractor.extract(pack, instance)
#                 feature_collection[tag] = feature
#             self.features_pools.append(feature_collection)
#             self.pack_pools[-1][1] += 1

#             if self.should_yield():
#                 yield self.create_batch()

#     def reamin_pack_num(self):
#         return len(self.pack_pools)

#     def should_yield(self):
#         return len(self.features_pools) >= self.batch_size

#     def create_batch(self, flush=False):
#         if flush:
#             pack_info = self.pack_pools
#             self.pack_pools = []
#         else:
#             pack_info = []
#             remain = self.batch_size

#             for i in range(len(self.pack_pools)):
#                 if remain - self.pack_pools[i][1] >= self.batch_size:
#                     remain -= self.pack_pools[0][1]
#                     pack_info.append(self.pack_pools.pop(0))
#                     if remain == 0:
#                         break
#                 else:
#                     pack_info.append(copy(self.pack_pools[0]))
#                     pack_info[-1][1] = remain
#                     self.pack_pools[0][1] -= remain

#         example_collection = {}
#         for features_collection in self.features_pools[:self.batch_size]:
#             for tag, feature in features_collection.items():
#                 if tag not in example_collection:
#                     example_collection[tag] = []
#                 example_collection[tag].append(feature)

#         tensor_collection = {}
#         for tag, features in example_collection.items():
#             converter = self.feature_scheme[tag]["converter"]
#             tensor, mask = converter.convert(features)

#             if tag not in tensor_collection:
#                 tensor_collection[tag] = {}

#             tensor_collection[tag]["tensor"] = tensor
#             tensor_collection[tag]["mask"] = mask

#             return tensor_collection, pack_info


# class Batcher(Generic[PackType]):

#     def __init__(self, cross_pack: bool = True):
#         self.current_batch: Dict = {}
#         self.data_pack_pool: List[PackType] = []
#         self.current_batch_sources: List[int] = []

#         self.cross_pack: bool = cross_pack

#     def initialize(self, batch_size):
#         self.batch_size = batch_size
#         self.current_batch.clear()
#         self.data_pack_pool.clear()
#         self.current_batch_sources.clear()


#     def flush(self) -> Iterator[Dict]:
#         if self.current_batch:
#             yield self.current_batch
#             self.current_batch = {}
#             self.current_batch_sources = []

#     def get_batch(
#             self, input_pack: PackType, context_type: Type[Annotation],
#             requests: DataRequest) -> Iterator[Dict]:
#         r"""Returns an iterator of data batches."""
#         # cache the new pack and generate batches
#         self.data_pack_pool.append(input_pack)

#         for (data_batch, instance_num) in self._get_data_batch(
#                 input_pack, context_type, requests):
#             self.current_batch = merge_batches(
#                 [self.current_batch, data_batch])
#             self.current_batch_sources.append(instance_num)

#             # Yield a batch on two conditions.
#             # 1. If we do not want to have batches from different pack, we
#             # should yield since this pack is exhausted.
#             # 2. We should also yield when the batcher condition is met:
#             # i.e. ``_should_yield()`` is True.
#             if not self.cross_pack or self._should_yield():
#                 yield self.current_batch
#                 self.current_batch = {}
#                 self.current_batch_sources = []

#     def _should_yield(self) -> bool:
#         return self.batch_is_full

#     def _get_data_batch(
#             self, data_pack: DataPack, context_type: Type[Annotation],
#             requests: Optional[Dict[Type[Entry], Union[Dict, List]]] = None,
#             offset: int = 0) -> Iterable[Tuple[Dict, int]]:
#         instances: List[Dict] = []
#         current_size = sum(self.current_batch_sources)

#         for data in data_pack.get_data(context_type, requests, offset):
#             instances.append(data)
#             if len(instances) == self.batch_size - current_size:
#                 batch = batch_instances(instances)
#                 self.batch_is_full = True
#                 yield (batch, len(instances))
#                 instances = []
#                 self.batch_is_full = False

#         # Flush the remaining data.
#         if len(instances) > 0:
#             batch = batch_instances(instances)
#             yield (batch, len(instances))
