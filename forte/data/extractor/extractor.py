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
import numpy as np
from torch import Tensor
from typing import Iterator, Dict, List, Any, Union, Set, Tuple

from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Token, Sentence, Document, Annotation
from forte.data.ontology.core import EntryType
from forte.data.span import Span
from forte.data.extractor.vocabulary import Vocabulary


class BaseExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.entry = config["entry"]
        use_pad = config.get("vocab_use_pad", False)
        use_unk = config.get("vocab_use_unk", False)
        method = config.get("vocab_method", "indexing")
        self.__vocab = Vocabulary(method = method,
                                use_pad = use_pad,
                                use_unk = use_unk)

    def size(self):
        return self.__vocab.size()

    def contains(self, entry: Any):
        return self.__vocab.contians(entry)

    def items(self):
        return self.__vocab.items()

    def add_entry(self, entry):
        self.__vocab.add_entry(entry)

    def entry2id(self, entry):
        return self.__vocab.entry2id(entry)

    def id2entry(self, idx):
        return self.__vocab.id2entry(idx)

    def get_pad_id(self):
        return self.__vocab.get_pad_id()

    def update_vocab(self, pack: DataPack, instance: EntryType):
        raise NotImplementedError()

    def extract(self, pack: DataPack, 
            instance: EntryType) -> Tensor:
        raise NotImplementedError() 

    def add_to_pack(self, pack: DataPack, instance: EntryType, tensor: Tensor):
        raise NotImplementedError()


class AttributeExtractor(BaseExtractor):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.attribute = config["attribute"]

    def update_vocab(self, pack: DataPack, instance: EntryType):
        for entry in pack.get(self.entry, instance):
            self.add_entry(getattr(entry, self.attribute))

    def extract(self, pack: DataPack, instance: EntryType):
        tensor = []
        for entry in pack.get(self.entry, instance):
            tensor.append(self.entry2id(getattr(entry, self.attribute)))
        return Tensor(tensor)

    # def add_to_pack(self, pack: DataPack, instance: EntryType, tensor: Tensor):
    #     for entry, idx in zip(pack.get(self.entry, instance),
    #                             tensor.numpy()):
    #         setattr(entry, self.attribute, self.id2entry(idx))
            

class TextExtractor(AttributeExtractor):
    def __init__(self, config: Dict):
        config["attribute"] = "text"
        super().__init__(config)


class CharExtractor(BaseExtractor):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_char_length = getattr(config, "max_char_length", None)

    def update_vocab(self, pack: DataPack, instance: EntryType):
        for word in pack.get(self.entry, instance):
            for char in word.text:
                self.add_entry(char)

    def extract(self, pack: DataPack, instance: EntryType) -> Tuple[Tensor, Tensor]:
        tensor = []
        mask = []
        max_char_length = -1

        for word in pack.get(self.entry, instance):
            tmp = []
            for char in word.text:
                tmp.append(self.entry2id(char))
            tensor.append(tmp)
            max_char_length = max(max_char_length, len(tmp))

        if self.max_char_length is not None:
            max_char_length = min(self.max_char_length, max_char_length)

        for i in range(len(tensor)):
            if len(tensor[i]) >= max_char_length:
                tensor[i] = tensor[i][:max_char_length]
                mask.append(np.ones_like(tensor[i], dtype = np.int))
            else:
                mask.append(np.concatenate([np.ones_like(tensor[i], dtype = np.int),
                                            np.zeros(shape = [max_char_length-len(tensor[i])],
                                            dtype = np.int)]))
                tensor[i] = tensor[i]+\
                    [self.get_pad_id()]*(max_char_length-len(tensor[i]))
                    
        return Tensor(tensor), Tensor(mask)



class AnnotationSeqExtractor(BaseExtractor):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.entry = config["entry"]
        self.attribute = config["attribute"]
        self.strategy = config["strategy"]
        self.based_on = config["based_on"]

    @classmethod
    def bio_variance(cls, tag):
        return [(tag, "B"), (tag, "I"), (None, "O")]

    @classmethod
    def bio_tag(cls, instance_based_on, instance_entry):
        tagged = []
        cur_entry_id = 0
        prev_entry_id = None
        cur_based_on_id = 0
        while cur_based_on_id < len(instance_based_on):
            base_begin = instance_based_on[cur_based_on_id].begin
            base_end = instance_based_on[cur_based_on_id].end

            if cur_entry_id < len(instance_entry):
                entry_begin = instance_entry[cur_entry_id].begin
                entry_end = instance_entry[cur_entry_id].end
            else:
                lastone = len(instance_based_on) - 1
                entry_begin = instance_based_on[lastone].end + 1
                entry_end = instance_based_on[lastone].end + 1


            if base_end < entry_begin:
                # Base: [...]
                # Entry       [....]
                tagged.append((None, "O"))
                prev_entry_id = None
                cur_based_on_id += 1
            elif base_begin >= entry_begin and base_end <= entry_end:
                # Base:    [...]
                # Entry:  [.......]
                if prev_entry_id == cur_entry_id:
                    tagged.append((instance_entry[cur_entry_id], "I"))
                else:
                    tagged.append((instance_entry[cur_entry_id], "B"))
                prev_entry_id = cur_entry_id
                cur_entry_id += 1
            elif base_begin > entry_end:
                # Base:         [...]
                # Entry: [....]
                cur_entry_id += 1
            else:
                raise AssertionError("Unconsidered case.")

        return tagged

    def update_vocab(self, pack: DataPack, instance: EntryType):
        for entry in pack.get(self.entry, instance):
            attribute = getattr(entry, self.attribute)
            for tag_variance in self.bio_variance(attribute):
                self.add_entry(tag_variance)

    def extract(self, pack: DataPack, instance: EntryType):
        instance_based_on = list(pack.get(self.based_on, instance))
        instance_entry = list(pack.get(self.entry, instance))
        instance_tagged = self.bio_tag(instance_based_on, instance_entry)
        ans = []
        for pair in instance_tagged:
            if pair[0] is None:
                new_pair = (None, pair[1])
            else:
                new_pair = (getattr(pair[0], self.attribute), pair[1])
            ans.append(self.entry2id(new_pair))
        return Tensor(ans)

    # def add_to_pack(self, pack: DataPack, instance: EntryType, tensor: Tensor):
    #     for entry, idx in zip(pack.get(self.entry, instance), tensor.numpy()):
    #         tag = self.id2entry(idx)
    #         setattr(entry, self.attribute, tag[0])
