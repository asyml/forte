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

import torch

import texar.torch as tx

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Sentence


__all__ = [
    "BERTEmbedder",
]


class BERTEmbedder(PackProcessor):
    r"""A wrapper of Texar :class:`BERTEncoder`.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.encoder = None
        self.sentence_component = None

    def initialize(self, resource: Resources, configs: HParams):
        self.tokenizer = tx.data.BERTTokenizer(
            pretrained_model_name=configs.pretrained_model_name)
        self.encoder = tx.modules.BERTEncoder(
            pretrained_model_name=configs.pretrained_model_name)

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(entry_type=Sentence,
                                       component=self.sentence_component):
            input_ids, segment_ids, _ = self.tokenizer.encode_text(
                text_a=sentence.text)

            input_ids = torch.tensor([input_ids])
            segment_ids = torch.tensor([segment_ids])
            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            output, _ = self.encoder(input_ids, input_length, segment_ids)
            sentence.embedding = output.tolist()

    @staticmethod
    def default_configs():
        r"""This default configurations for :class:`BERTTokenizer`.
        """
        return {
            'pretrained_model_name': 'bert-base-uncased',
        }
