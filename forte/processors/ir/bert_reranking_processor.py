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

from texar.torch.data.tokenizers.bert_tokenizer import BERTTokenizer

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Query
from forte.processors.base import MultiPackProcessor

from examples.passage_ranker.bert import (
    BERTClassifier, BERTEncoder)

__all__ = [
    "BertRerankingProcessor"
]


class BertRerankingProcessor(MultiPackProcessor):

    def initialize(self, resources: Resources, configs: Config):
        self.resources = resources
        self.config = Config(configs, self.default_configs())

        # TODO: At the time of writing, no way in texar to set encoder in
        # `texar.torch.modules.classifiers.BertClassifier`. Should not ideally
        # be changing a private variable.
        # pylint: disable=protected-access
        BERTClassifier._ENCODER_CLASS = BERTEncoder
        # pylint: enable=protected-access

        cache_dir = os.path.join(os.path.dirname(__file__),
                                 self.config.model_dir)

        self.device = torch.device('cuda:0') \
            if torch.cuda.is_available() else torch.device('cpu')

        self.model = BERTClassifier(
            pretrained_model_name=self.config.pretrained_model_name,
            cache_dir=cache_dir,
            hparams=self.config).to(self.device)

        self.tokenizer = BERTTokenizer(
            pretrained_model_name=self.config.pretrained_model_name,
            cache_dir=cache_dir,
            hparams=None)

    @staticmethod
    def default_configs() -> Dict[str, Any]:
        pretrained_model_name = "bert-large-uncased"
        return {
            "size": 5,
            "query_pack_name": "query",
            "field": "content",
            "pretrained_model_name": pretrained_model_name,
            "model_dir": os.path.join(os.path.dirname(__file__), "models"),
            "max_seq_length": 512
        }

    def _process(self, input_pack: MultiPack):
        max_len = self.config.max_seq_length
        query_pack_name = self.config.query_pack_name

        query_pack = input_pack.get_pack(self.config.query_pack_name)
        query_entry = list(query_pack.get(Query))[0]
        query_text = query_pack.text

        packs = {}
        for doc_id in input_pack.pack_names:
            if doc_id == query_pack_name:
                continue

            pack = input_pack.get_pack(doc_id)
            document_text = pack.text

            # BERT Inference
            input_ids, segment_ids, input_mask = [
                torch.LongTensor(item).unsqueeze(0).to(self.device)
                for item in self.tokenizer.encode_text(
                    query_text, document_text, max_len)]

            seq_length = (input_mask == 1).sum(dim=-1)
            logits, _ = self.model(input_ids, seq_length, segment_ids)
            preds = torch.nn.functional.softmax(torch.Tensor(logits), dim=1)

            score = preds.detach().tolist()[0][1]

            query_entry.update_results({doc_id: score})
            packs[doc_id] = pack
