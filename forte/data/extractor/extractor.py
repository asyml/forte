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
from typing import Iterator, Dict, List, Any, Union, Set

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Token, Sentence, Document, Annotation
from forte.data.ontology.core import EntryType
from forte.data.span import Span
from forte.data.converter.vocabulary import Vocabulary
from torch import Tensor
from abc import abstractmethod


'''
Training:
for tag in data_request:
    converters[tag] = Converter(config)

raw: PER , LOC 
enc dat: B-PER, I-PER
tensor: 1,2,5

# TODO: explictiy build vocab
# BIO-NER etc.

# TODO: Implicity build vocab (without caching enc_data) (remove rare words etc.)
for pack in datapacks:
    for tag in data_request: file1 2... 10
        convertes[tag].consume_instance(pack, scope)


merge together{
# TODO: cache enc_data
for pack in datapacks:
    for tag in data_request: 
        convertes[tag].consume_instance(pack, scope)

# TODO: loop over all data (use cached data or from raw data)
for tag in data_requests:
    tensors[tag] = converters[tag].produce_tensors()
    model.train(tensors)
}

#TODO: cache machasin (better to use other package, pytorch, pytorch-lightening, tensorflow)
1. memory base
2. file: file1, file2

save(conveters)

------------------------
Prediction:
# TODO: seperate conveters into Features and Output part
for tag in data_request_not_predict:
    converters[tag].set_eval(to_predict=False)
for tag in data_request_to_predict:
    converters[tag].set_eval(to_predict=True)

for pack in datapacks:
    for instance in pack:
        for tag in data_request:
            convertes[tag].consume_instance(pack, instance)
    
    for tag in data_request_not_predict:
        tensors[tag] = converters[tag].produce_instance()
    for tag in data_request_to_predict:
        metadatas[tag] = converters[tag].produce_instance()

    outputs = model.predict(tensors)

    for tag in data_request_to_predict:
        coverters[tag].add_to_pack(pack, outputs[tag], metadatas[tag])
'''


class BaseExtractor:
    def __init__(self, config: Dict):
        self.vocab = Vocabulary()
        pass

    def init_vocab(self, labels:Union[List,Set,Dict]=None):
        pass

    @abstractmethod
    def update_vocab(self, pack: DataPack, instance: EntryType):
        raise NotImplementedError()

    def build_vocab(self):
        pass
    
    @abstractmethod
    def extract(self, pack: DataPack, 
            instance: EntryType) -> Tensor:
        raise NotImplementedError() 
    
    @abstractmethod
    def add_to_pack(self, pack: DataPack, instance: EntryType, tensor: Tensor):
        raise NotImplementedError()

class AttributeExtractor(BaseExtractor):
    def __init__(self, config: Dict):
        super().__init__(config)

    def update_vocab(self, pack: DataPack, instance: EntryType):


class TextExtractor(BaseExtractor):
    def __init__(self, config: Dict):

class CharExtractor(BaseExtractor):
    def __init__(self, config: Dict):

class AnnotationSeqConverter(BaseExtractor):
    def __init__(self, config: Dict):


# for pack in packs:
#     for instance in pack:
#         output = model.predict(extractor1.extrac(instance))

#         extractor2.add_to_pack(pack, instance, output)


# class BaseConverter:
#     def __init__(self, config: Dict):
#         self.mode = "train"
#         self.data = []

#     def set_eval(self, to_predict=False):
#         if to_predict:
#             self.mode = "eval_to_predict"
#             self.data = []
#         else:
#             self.mode = "eval"
#             self.data = []

#     def consume_instance(self, pack: DataPack, instance: Annotation):
#         raise NotImplementedError()

#     def produce_instance(self) -> List[Tensor]:
#         raise NotImplementedError()

#     def add_to_datapack(self, pack: DataPack, tensors: Tensor,
#                             meta_data: List[Span]) -> DataPack:
#         raise NotImplementedError()

# class OneToOneConverter(BaseConverter):
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.entry = config["entry"]
#         self.label = getattr(config, "label", "text")

#         # TODO: unify the name, type for "repr"
#         if config["repr"] == "text_repr":
#             self.conversion_method = lambda x: x
#         elif callable(config["repr"]):
#             self.conversion_method = config["repr"]
#         else:
#             raise NotImplementedError("Unknow repr.")

#         # TODO: if these are the only two methods
#         if config["conversion_method"] == "indexing":
#             self.vocab = Vocabulary(method = "indexing")
#         elif config["conversion_method"] == "one-hot":
#             self.vocab = Vocabulary(method = "one-hot")
#         else:
#             raise NotImplementedError("Unknow coversion method")

#     def consume_instance(self, pack: DataPack, instance: Annotation):
#         if self.mode == "train" or self.mode == "eval":
#             instance = []
#             for e in pack.get(self.entry, instance):
#                 e_converted = self.conversion_method(getattr(e, self.label))
#                 instance.append(e_converted)
#                 if self.mode == "train":
#                     self.vocab.add_entry(e_converted)
#             self.data.append(instance)
#         elif self.mode == "eval_to_predict":
#             instance = []
#             for e in pack.get(self.entry, instance):
#                 instance.append(e.span)
#             self.data.append(instance)
#         else:
#             raise NotImplementedError()

#     def produce_instance(self) -> List[Tensor]:
#         if self.mode == "train" or self.mode == "eval":
#             self.vocab.build()
#             ans = []
#             for instance in self.data:
#                 ans.append(map(self.vocab.to_id, instance))
#         elif self.mode == "eval_to_predict":
#             ans = self.data
#         return ans

#     def add_to_datapack(self, pack: DataPack, tensors: Tensor,
#                             meta_data: List[Span]=None) -> DataPack:
#         if meta_data is None:
#             meta_data = self.data

#         for tensor, span in zip(tensors.numpy(), meta_data):
#             entry = self.entry(pack, span.begin, span.end)
#             setattr(entry, self.label, self.vocab.from_id(int(tensor)))

# # AttributeConverter
# # TextCoverter 
# # CharCoverter
# # AnnotationSeqConverter:  Two based on, annotation -> fn(basedon, annotion, attri)
# # TODO: simpler name

# class OneToManyConverter(BaseConverter):
#     # should we expand many or keep as a list
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.entry = config["entry"]
#         self.label = getattr(config, "label", "text")

#         if config["repr"] == "char_repr":
#             self.conversion_method = lambda x: x.split()
#         elif callable(config["repr"]):
#             self.conversion_method = config["repr"]
#         else:
#             raise NotImplementedError("Unknow repr.")
        
#         if config["conversion_method"] == "indexing":
#             self.vocab = Vocabulary(method = "indexing")
#         elif config["conversion_method"] == "one-hot":
#             self.vocab = Vocabulary(method = "one-hot")
#         else:
#             raise NotImplementedError("Unknow coversion method")

#     def consume_instance(self, pack: DataPack, instance: Annotation):
#         if self.mode == "train" or self.mode == "eval":
#             instance = []
#             for e in pack.get(self.entry, instance):
#                 e_converted = self.conversion_method(getattr(e, self.label))
#                 instance.append(e_converted)
#                 if self.mode == "train":
#                     for component in e_converted:
#                         self.vocab.add_entry(component)
#             self.data.append(instance)
#         elif self.mode == "eval_to_predict":
#             instance = []
#             for e in pack.get(self.entry, instance):
#                 instance.append(e.span)
#             self.data.append(instance)
#         else:
#             raise NotImplementedError()

#     def produce_instance(self) -> List[Tensor]:
#         if self.mode == "train" or self.mode == "eval":
#             self.vocab.build()
#             ans = []
#             for instance in self.data:
#                 ans.append([map(self.vocab.to_id, entry) for entry in instance])
#         elif self.mode == "eval_to_predict":
#             ans = self.data
#         return ans

#     def add_to_datapack(self, pack: DataPack, tensors: Tensor,
#                             meta_data: List[Span]=None) -> DataPack:
#         if meta_data is None:
#             meta_data = self.data
             
#         # TODO: should we assume [[a b c] [a b c d e]] in tensor (list of tensors?)
#         for tensor, span in zip(tensors, meta_data):
#             #TODO: what type to use create the sub component?
#             for i, component in enumerate(tensor):
#                 component = component.numpy()
#                 entry = self.entry(pack, span.begin+i, span.begin+i+1)
#             setattr(entry, self.label, self.vocab.from_id(int(tensor)))


# class ManyToOneConverter(BaseConverter):
#     def __init__(self, config: Dict):
#         super().__init__(config)
#         self.entry = config["entry"] # ANTYMENTION
#         self.label = getattr(config, "label", "text") # ner_type
#         self.strategy = config["strategy"]

#         # TODO: do we really need "based on" when doing
#         # many to one?

#         if config["conversion_method"] == "indexing":
#             self.vocab = Vocabulary(method = "indexing")
#         elif config["conversion_method"] == "one-hot":
#             self.vocab = Vocabulary(method = "one-hot")
#         else:
#             raise NotImplementedError("Unknow coversion method")

#     def consume_instance(self, pack:DataPack, instance: Annotation):
#         if self.strategy == "BIO":
#             return self.consume_instance_bio(pack, instance)
#         else:
#             raise NotImplementedError()

#     def consume_instance_bio(self, pack: DataPack, instance: Annotation): #SENTCNE
#         if self.mode == "train":
#             instance = []
#             for e in pack.get(self.entry, instance):
#                 instance = getattr(e, self.label)
#             # Based on: [TOKEN1, TOKEN2 TOKEN3]
#             # LABEL: [PER ] ANNTITI.span

#             converted = []
#             if len(instance) == 0:
#                 pass
#             elif len(instance) == 1:
#                 converted.append((instance[0], 'B'))
#             else:
#                 prev_label = None
#                 for cur_label in instance:
#                     if cur_label == prev_label:
#                         if prev_label == 'O':
#                             converted.append((cur_label, 'O'))
#                         else:
#                             converted.append((cur_label, 'I'))
#                         prev_label = cur_label
#                     elif cur_label == 'O':
#                         converted.append((cur_label, 'O'))
#                         prev_label = cur_label
#                     else:
#                         converted.append((cur_label, 'B'))
#                         prev_label = cur_label
            
#             for component in converted:
#                 self.vocab.add_entry(component)


        
#         # elif self.mode == "eval":
#         #     instance = []
#         #     for e in pack.get(self.entry, instance):
#         #         e_converted = self.conversion_method(getattr(e, self.label))
#         #         instance.append(e_converted)
#         #     self.data.append(instance)
#         # elif self.mode == "eval_to_predict":
#         #     instance = []
#         #     for e in pack.get(self.entry, instance):
#         #         instance.append(e.span)
#         #     self.data.append(instance)
#         # else:
#         #     raise NotImplementedError()

#     # def produce_instance(self) -> List[Tensor]:
#     #     if self.mode == "train":
#     #         self.vocab.build()
#     #         ans = []
#     #         for instance in self.data:
#     #             ans.append([map(self.vocab.to_id, entry) for entry in instance])
#     #     elif self.mode == "eval":
#     #         ans = []
#     #         for instance in self.data:
#     #             ans.append([map(self.vocab.to_id, entry) for entry in instance])
#     #     elif self.mode == "eval_to_predict":
#     #         ans = self.data
#     #     return ans

#     # def add_to_datapack(self, pack: DataPack, tensors: Tensor,
#     #                         meta_data: List[Span]=None) -> DataPack:
#     #     if meta_data is None:
#     #         meta_data = self.data

#     #     # TODO: should we assume [[a b c] [a b c d e]] in tensor (list of tensors?)
#     #     for tensor, span in zip(tensors, meta_data):
#     #         #TODO: what type to use create the sub component?
#     #         for i, component in enumerate(tensor):
#     #             component = component.numpy()
#     #             entry = self.entry(pack, span.begin+i, span.begin+i+1)
#     #         setattr(entry, self.label, self.vocab.from_id(int(tensor)))

