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
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.data.types import DataRequest
from forte.models.ner import utils
from forte.data.base_pack import PackType
from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.processors.base.batch_processor import FixedSizeBatchProcessor
from forte.processors.base.base_processor import BaseProcessor
from ft.onto.base_ontology import Token, Sentence, EntityMention
from forte.data.extractor.extractor import TextExtractor, AnnotationSeqExtractor

logger = logging.getLogger(__name__)


class Predictor(BaseProcessor):
    """
       An Named Entity Recognizer trained according to `Ma, Xuezhe, and Eduard
       Hovy. "End-to-end sequence labeling via bi-directional lstm-cnns-crf."
       <https://arxiv.org/abs/1603.01354>`_.

       Note that to use :class:`CoNLLNERPredictor`, the :attr:`ontology` of
       :class:`Pipeline` must be an ontology that include
       ``ft.onto.base_ontology.Token`` and ``ft.onto.base_ontology.Sentence``.
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.word_alphabet, self.char_alphabet, self.ner_alphabet = (
            None, None, None)
        self.resource = None

    def initialize(self, resource: Resources, configs: Config):
        self.resource = resource
        self.configs = configs
        print("configs", configs)
        print("resource", resource.resources)
        self.extractors = self.load_extractors(configs)
        self.model = self.load_model(configs)

    def load_extractors(self, configs):
        config1 = {
            "scope": Sentence,
            "entry": EntityMention,
            "attribute": "ner_type",
            "based_on": Token,
            "strategy": "BIO",
            "vocab_use_unk": True
        }
        extractor1 = AnnotationSeqExtractor(config1)
        extractor1.add_entry(("PER", "B"))
        config0 = {
            "scope": Sentence,
            "entry": Token,
            "vocab_use_unk": True
        }
        extractor0 = TextExtractor(config0)
        return [extractor0, extractor1]

    def load_model(self, configs):
        class Model:
            def __call__(self, tensor):
                # Input shape: [Batch, length]
                # Output shape: [Batch]
                batch_size = len(tensor)
                length = len(tensor[0])
                return [[2 for _ in range(length)] for _ in range(batch_size)]
        return Model()


    def new_pack(self, pack_name: Optional[str] = None) -> DataPack:
        return DataPack(pack_name)

    def _process(self, input_pack: PackType):
        for instance in input_pack.get(Sentence):
            feat0 = self.extractors[0].extract(input_pack, instance)
            feat1 = self.extractors[1].extract(input_pack, instance)
            input_tensor = [feat0.data] # add batch dim
            output_tensor = self.model(input_tensor)[0] # extract batch dim
            feat1.data = output_tensor
            self.extractors[1].add_to_pack(input_pack, instance, feat1)
        return input_pack

    @classmethod
    def default_configs(cls):
        r"""Default config for NER Predictor"""

        configs = super().default_configs()
        # TODO: Batcher in NER need to be update to use the sytem one.

        more_configs = {
            "config_data": {
                "train_path": "",
                "val_path": "",
                "test_path": "",
                "num_epochs": 200,
                "batch_size_tokens": 512,
                "test_batch_size": 16,
                "max_char_length": 45,
                "num_char_pad": 2
            },
        }

        configs.update(more_configs)
        return configs
