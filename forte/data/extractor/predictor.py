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
                                        tensor_collection["text_tag"]["mask"][0],
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
