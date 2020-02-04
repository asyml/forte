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
from forte.data.ontology.top import Annotation
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.utils.utils import get_class


__all__ = [
    "PretrainedEncoder",
]


class PretrainedEncoder(PackProcessor):
    r"""A wrapper of Texar pre-trained encoders.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.encoder = None
        self.entry_type = None

    # pylint: disable=unused-argument
    def initialize(self, resource: Resources, configs: HParams):
        if configs.pretrained_model_name.startswith('bert'):
            self.tokenizer = tx.data.BERTTokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.BERTEncoder(
                pretrained_model_name=configs.pretrained_model_name)
        elif configs.pretrained_model_name.startswith('gpt2'):
            self.tokenizer = tx.data.GPT2Tokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.GPT2Encoder(
                pretrained_model_name=configs.pretrained_model_name)
        elif configs.pretrained_model_name.startswith('roberta'):
            self.tokenizer = tx.data.RoBERTaTokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.RoBERTaEncoder(
                pretrained_model_name=configs.pretrained_model_name)
        elif configs.pretrained_model_name.startswith('T5'):
            self.tokenizer = tx.data.T5Tokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.T5Encoder(
                pretrained_model_name=configs.pretrained_model_name)
        elif configs.pretrained_model_name.startswith('xlnet'):
            self.tokenizer = tx.data.XLNetTokenizer(
                pretrained_model_name=configs.pretrained_model_name)
            self.encoder = tx.modules.XLNetEncoder(
                pretrained_model_name=configs.pretrained_model_name)
        else:
            raise ValueError("Unrecognized pre-trained model name.")

        self.entry_type = get_class(configs.entry_type)
        if not isinstance(self.entry_type, Annotation) and \
                not issubclass(self.entry_type, Annotation):
            raise ValueError("entry_type must be annotation type.")

    def _process(self, input_pack: DataPack):
        for entry in input_pack.get(entry_type=self.entry_type):
            input_ids, segment_ids, _ = self.tokenizer.encode_text(
                text_a=entry.text)

            input_ids = torch.tensor([input_ids])
            segment_ids = torch.tensor([segment_ids])
            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            output, _ = self.encoder(input_ids, input_length, segment_ids)
            entry.embedding = output.tolist()

    @staticmethod
    def default_configs():
        r"""This default configurations for :class:`PretrainedEncoder`.
        """
        return {
            'pretrained_model_name': 'bert-base-uncased',
            'entry_type': 'ft.onto.base_ontology.Sentence',
        }
