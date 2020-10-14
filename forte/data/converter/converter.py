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

import codecs
import logging
import os
from typing import Iterator, Dict, List, Any

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Token, Sentence, Document, Annotation
from forte.data import Span
from forte.data.converter.vocabulary import Vocabulary
from torch import Tensor


'''
Training:
for tag in data_request:
    converters[tag] = BaseConverter(config)

for pack in datapacks:
    for tag in data_request:
        convertes[tag].consume_instance(pack, scope)

for tag in data_requests:
    tensors[tag] = converters[tag].produce_tensors()

model.train(tensors)
save(conveters)

Prediction:
for tag in data_request:
    converters[tag].set_eval(to_predict=True)
    converters[tag].set_eval(to_predict=False)

for pack in datapacks:
    
    for instance in pack:
        for tag in data_request:
            convertes[tag].consume_instance(pack, scope)
    
    for tag in data_request_not_predict:
        tensors[tag] = converters[tag].produce_instance()
    
    for tag in data_request_to_predict:
        metadatas[tag] = converters[tag].produce_instance()

    outputs = model.predict(tensors)

    for tag in data_request_to_predict:
        coverters[tag].tensor2datapack(pack, outputs[tag], metadatas[tag])




'''


class BaseConverter:
    def __init__(self, config: Dict):
        self.entry = config["entry"]
        self.scope = config["scope"]
        self.field = getattr(config, "field", "text")
        self.conversion_method = config["conversion_method"]

        self.mode = "train"
        self.data = []
        self.vocab = Vocabulary()

    def set_eval(self, to_predict=False):
        if to_predict:
            self.mode = "eval_to_predict"
            self.data = []
        else:
            self.mode = "eval"
            self.data = []

    def consume_instance(self, pack: DataPack, range_annotation: Annotation):
        raise NotImplementedError()
   
    def produce_instance(self) -> List[Tensor]:
        raise NotImplementedError()

    def add_to_datapack(self, pack: DataPack, tensors: Tensor,
                            meta_data: List[Span]) -> DataPack:
        raise NotImplementedError()

class OneToOneConverter(BaseConverter):
    def __init__(self, config: Dict):
        super().__init__(config)
        

    def consume_instance(self, pack: DataPack, range_annotation: Annotation):
        if self.mode == "train":
            instance = []
            for e in pack.get(self.entry, range_annotation):
                e_converted = self.conversion_method(getattr(e, self.field))
                instance.append(e_converted)
                self.vocab.add_entry(e_converted)
            self.data.append(instance)
        elif self.mode == "eval":
            instance = []
            for e in pack.get(self.entry, range_annotation):
                e_converted = self.conversion_method(getattr(e, self.field))
                instance.append(e_converted)
            self.data.append(instance)
        elif self.mode == "eval_to_predict":
            instance = []
            for e in pack.get(self.entry, range_annotation):
                instance.append(e.span)
            self.data.append(instance)

    def produce_instance(self) -> List[Tensor]:
        if self.mode == "train":
            self.vocab.build()
            ans = []
            for instance in self.data:
                ans.append(map(self.vocab.to_id, instance))
        elif self.mode == "eval":
            ans = []
            for instance in self.data:
                ans.append(map(self.vocab.to_id, instance))
        elif self.mode == "eval":
            ans = self.data
        return ans

    def add_to_datapack(self, pack: DataPack, tensors: Tensor,
                            meta_data: List[Span]) -> DataPack:
        for tensor, span in zip(tensors.numpy(), meta_data):
            entry = self.entry(pack, span.begin, span.end)
            setattr(entry, self.field, self.vocab.from_id(int(tensor)))


class OneToManyConverter(BaseConverter):
    def __init__(self, config: Dict):
        super().__init__(config)


class ManyToOneConverter(BaseConverter):
    def __init__(self, config: Dict):
        super().__init__(config)

