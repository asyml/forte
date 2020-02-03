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

import texar.torch as tx

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Sentence, Subword


__all__ = [
    "BERTTokenizer",
]


class BERTTokenizer(PackProcessor):
    r"""A wrapper of Texar BERTTokenizer.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.sentence_component = None

    def initialize(self, resource: Resources, configs: HParams):
        self.tokenizer = tx.data.BERTTokenizer(
            pretrained_model_name=configs.pretrained_model_name)

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(entry_type=Sentence,
                                       component=self.sentence_component):
            offset = sentence.span.begin
            end_pos = 0
            for word in self.tokenizer.map_text_to_token(sentence.text):
                if word.startswith("##"):
                    word = word[2:]
                    begin_pos = end_pos
                    end_pos = begin_pos + len(word)
                else:
                    begin_pos = sentence.text.find(word, end_pos)
                    if begin_pos == -1:
                        begin_pos = sentence.text.lower().find(word, end_pos)
                    end_pos = begin_pos + len(word)
                subword = Subword(input_pack, begin_pos + offset,
                                  end_pos + offset)
                input_pack.add_or_get_entry(subword)

    @staticmethod
    def default_configs():
        r"""This default configurations for :class:`BERTTokenizer`.
        """
        return {
            'pretrained_model_name': 'bert-base-uncased',
        }
