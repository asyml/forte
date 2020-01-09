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
# pylint: disable=attribute-defined-outside-init
import os
from typing import Dict, Any

import torch

from texar.torch.hyperparams import HParams
from texar.torch.data.tokenizers.bert_tokenizer import BERTTokenizer

from forte.data import DataPack
from forte.common.resources import Resources
from forte.processors.base import RerankingProcessor
from forte.models.bert_ngyugen2019 import (
    FineTunedBERTClassifier, FineTunedBERTEncoder)

__all__ = [
    "BERTBasedRerankingProcessor"
]


class BERTBasedRerankingProcessor(RerankingProcessor):

    def initialize(self, resources: Resources, configs: HParams):
        super(BERTBasedRerankingProcessor, self).initialize(resources, configs)

        FineTunedBERTClassifier._ENCODER_CLASS = FineTunedBERTEncoder

        cache_dir = os.path.join(os.path.dirname(__file__),
                                 self.config.cache_dir)

        self.device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

        self.model = FineTunedBERTClassifier(
            pretrained_model_name=self.config.pretrained_model_name,
            cache_dir=cache_dir,
            hparams=self.config).to(self.device)

        self.tokenizer = BERTTokenizer(
            pretrained_model_name=self.config.pretrained_model_name,
            cache_dir=cache_dir,
            hparams=None)

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        pretrained_model_name = "bert-large-uncased"
        return {
            "size": 5,
            "query_pack_name": "query",
            "field": "content",
            "pretrained_model_name": pretrained_model_name,
            "cache_dir": os.path.join(os.path.dirname(__file__), "models"),
            "max_seq_length": 512
        }

    def get_matching_score(self, query_pack: DataPack, document_pack: DataPack):
        max_len = self.config.max_seq_length

        query_text = query_pack.text

        document_text = document_pack.text

        input_ids, segment_ids, input_mask = [
            torch.LongTensor(item).unsqueeze(0).to(self.device)
            for item in self.tokenizer.encode_text(
                query_text, document_text, max_len)]

        seq_length = (input_mask == 1).sum(dim=-1)
        logits, _ = self.model(input_ids, seq_length, segment_ids)
        preds = torch.nn.functional.softmax(torch.Tensor(logits), dim=1)

        score = preds.detach().tolist()[0][1]
        return score
