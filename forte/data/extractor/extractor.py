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
import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Iterator, Dict, List, Any, Union, Set, Tuple, Union

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.data_utils_io import dataset_path_iterator
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import Token, Sentence, Document, Annotation, EntityMention
from forte.data.ontology.core import EntryType
from forte.data.span import Span
from forte.data.extractor.vocabulary import Vocabulary
from forte.data.extractor.feature import Feature


class BaseExtractor(ABC):
    '''This class is used to get feature from the datapack and also
    add prediction back to datapack.
    '''
    def __init__(self, config: Union[Dict, Config]):
        '''Config will need to contains some value to initialize the
        extractor.
        Entry_type: Type[EntryType], every ectractor will get feature by loop on
            one type of entry in the instance. e.g. Token, EntityMention.
        Vocab_use_pad, Vocab_use_unk, Vocab_method" are used to configurate the
                vocabulary class.
        Vocab_predefined: a set of elements needed to be added to the vocabulary.
        '''
        defaults = {
            "entry_type": None,
            "vocab_use_pad": True,
            "vocab_use_unk": False,
            "vocab_method": "indexing",
            "vocab_predefined": set()
            }
        self.config = Config(config, default_hparams = defaults,
                                    allow_new_hparam = True)

        if self.config.entry_type is None:
            raise AttributeError("Entry_type is needed in the config.")
        print(self.config.vocab_use_pad)
        self.vocab = Vocabulary(method = self.config.vocab_method,
                                use_pad = self.config.vocab_use_pad,
                                use_unk = self.config.vocab_use_unk,
                                predefined = self.config.vocab_predefined)

    @abstractmethod
    def update_vocab(self, pack: DataPack, instance: Annotation):
        '''This function is used when user want to add element to vocabulary
        using the current instance. e.g. add all tokens in one sentence to
        the vocabulary.
        '''
        raise NotImplementedError()

    @abstractmethod
    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        '''This function will extract feature from one instance in the pack.
        '''
        raise NotImplementedError()

    def add_to_pack(self, pack: DataPack, instance: Annotation, prediction: Any):
        '''This function will add prediction to the pack according to different
        type of extractor.
        '''
        raise NotImplementedError()




class AttributeExtractor(BaseExtractor):
    '''This type of extractor will get the attribute on entry_type
    within one instance.
    '''    
    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)
        defaults = {
            "attribute": None,
        }
        self.config = Config(self.config, default_hparams = defaults,
                                            allow_new_hparam = True)
        if self.config.attribute is None:
            raise AttributeError("Attribute is needed for AttributeExtractor.")

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for entry in pack.get(self.config.entry_type, instance):
            self.vocab.add(getattr(entry, self.config.attribute))

    def extract(self, pack: DataPack, instance: Annotation) -> Feature:
        '''The AttributeExtractor only extract one attribute for one entry
        in the instance. There for the output feature will have same number
        of attributes as entries in one instance.
        '''
        data = []
        for entry in pack.get(self.config.entry_type, instance):
            idx = self.vocab.element2id(getattr(entry, self.config.attribute))
            data.append(idx)
        # One attribute correspond to one entry, therefore the dim is 1.
        return Feature(data = data, pad_value = self.vocab.get_pad_id(), dim = 1)

    def add_to_pack(self, pack: DataPack, instance: Annotation, prediction: Any):
        attrs = [self.vocab.id2element(x) for x in prediction]
        for entry, attr in zip(pack.get(self.config.entry_type, instance), attrs):
            setattr(entry, self.config.attribute, attr)




class TextExtractor(AttributeExtractor):
    '''A special type of AttributeExtractor, TextExtractor.
    It extract the text attribute on entry within one instance.
    '''
    def __init__(self, config: Union[Dict, Config]):
        config["attribute"] = 'text'
        super().__init__(config)


class CharExtractor(BaseExtractor):
    '''CharExtractor will get each char for each token in the instance.'''
    def __init__(self, config: Union[Dict, Config]):
        defaults = {
                "max_char_length": None
            }
        super().__init__(config)
        self.config = Config(self.config,
                                default_hparams = defaults,
                                allow_new_hparam = True)

    def update_vocab(self, pack: DataPack, instance: EntryType):
        for word in pack.get(self.config.entry_type, instance):
            for char in word.text:
                self.vocab.add(char)

    def extract(self, pack: DataPack, instance: EntryType) -> Feature:
        data = []
        max_char_length = -1

        for word in pack.get(self.config.entry_type, instance):
            tmp = []
            for char in word.text:
                tmp.append(self.vocab.element2id(char))
            data.append(tmp)
            max_char_length = max(max_char_length, len(tmp))

        if self.config.max_char_length is not None:
            max_char_length = min(self.config.max_char_length,
                                    max_char_length)
        # For each token, the output is a list of characters.
        # Therefore the dim is 2.
        ans = Feature(data = data, pad_value = self.vocab.get_pad_id(),
                        dim = 2)
        return ans


class BioSeqTaggingExtractor(BaseExtractor):

    def __init__(self, config: Union[Dict, Config]):
        super().__init__(config)
        defaults = {
            "attribute": None,
            "strategy": None,
            "based_on": None
        }
        self.config = Config(self.config,
                                default_hparams = defaults,
                                allow_new_hparam = True)

    def bio_variance(self, tag):
        return [(tag, "B"), (tag, "I"), (None, "O")]

    def bio_tag(self, instance_based_on, instance_entry):
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
                cur_based_on_id += 1
            elif base_begin > entry_end:
                # Base:         [...]
                # Entry: [....]
                cur_entry_id += 1
            else:
                raise AssertionError("Unconsidered case.")
        return tagged

    def update_vocab(self, pack: DataPack, instance: Annotation):
        for entry in pack.get(self.config.entry_type, instance):
            attribute = getattr(entry, self.config.attribute)
            for tag_variance in self.bio_variance(attribute):
                self.vocab.add(tag_variance)

    def extract(self, pack: DataPack, instance: Annotation):
        instance_based_on = list(pack.get(self.config.based_on, instance))
        instance_entry = list(pack.get(self.config.entry_type, instance))
        instance_tagged = self.bio_tag(instance_based_on, instance_entry)

        data = []
        for pair in instance_tagged:
            if pair[0] is None:
                new_pair = (None, pair[1])
            else:
                new_pair = (getattr(pair[0], self.config.attribute), pair[1])
            data.append(self.vocab.element2id(new_pair))

        return Feature(data, self.vocab.get_pad_id(), 1)

    def add_to_pack(self, pack: DataPack, instance: EntryType, prediction: List):
        tags = [self.vocab.id2element(x) for x in prediction]
        tag_start = None
        tag_end = None
        tag_type = None
        cnt = 0
        for entry, tag in zip(pack.get(self.config.based_on, instance), tags):
            # A new tag occurs
            if tag[1] == "O" or tag[1] == "B":
                # Handle previous tag
                if tag_type:
                    entity_mention = EntityMention(pack, tag_start, tag_end)
                    entity_mention.ner_type = tag_type
                tag_start = entry.begin
                tag_end = entry.end
                tag_type = tag[0]
            # It is same tag
            else:
                # TODO: handle not current tag as output
                assert entry[0] == tag_type
                tag_end = entry.end
            cnt += 1

        # Handle the final tag
        if tag_type:
            entity_mention = EntityMention(pack, tag_start, tag_end)
            entity_mention.ner_type = tag_type

