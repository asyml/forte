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

from forte.data.types import DATA_INPUT, DATA_OUTPUT
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


class Batcher:
    def __init__(self, batch_size, feature_resource):
        self.batch_size = batch_size
        self.feature_resource = feature_resource
        self.pack_pools = []
        self.instance_pools = []
        self.features_pools = []

    def current_pack_len(self):
        return len(set(self.pack_pools))

    def convert(self, feature_collections):
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

    def yield_batch(self, pack):
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

    def flush_batch(self):
        if len(self.features_pools) > 0:
            yield self.convert(self.features_pools), self.pack_pools, \
                self.instance_pools
            self.features_pools.clear()
            self.pack_pools.clear()
            self.instance_pools.clear()


class Predictor(BaseProcessor):
    def __init__(self, batch_size, predict_foward_fn, feature_resource):
        super().__init__()
        self.feature_resource = feature_resource
        self.batcher = Batcher(batch_size = batch_size,
                                feature_resource = feature_resource)
        self.predict_foward_fn = predict_foward_fn

    def unpad(self, predictions, packs, instances):
        new_predictions = {}
        for tag, preds in predictions.items():
            new_preds = []
            for pred, pack, instance in zip(preds, packs, instances):
                new_preds.append(self.feature_resource['schemes'][tag]['unpadder'].unpad(
                    pred, pack, instance))
            new_predictions[tag] = new_preds
        return new_predictions

    def add_to_pack(self, predictions, packs, instances):
        for tag, preds in predictions.items():
            for pred, pack, instance in zip(preds, packs, instances):
                self.feature_resource['schemes'][tag]['extractor'].add_to_pack(
                            pack, instance, pred)
                pack.add_all_remaining_entries()

    def _process(self, input_pack: PackType):
        for tensor_collection, packs, instances in self.batcher.yield_batch(input_pack):
            predictions = self.predict_foward_fn(tensor_collection)
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
            predictions = self.predict_foward_fn(tensor_collection)
            predictions = self.unpad(predictions, packs, instances)
            self.add_to_pack(predictions, packs, instances)

        current_queue = self._process_manager.current_queue
        for job in current_queue:
            job.set_status(ProcessJobStatus.PROCESSED)


    def new_pack(self, pack_name: Optional[str] = None) -> DataPack:
        return DataPack(pack_name)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        super_config = super().default_configs()
        return super_config
