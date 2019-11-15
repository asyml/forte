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
import torch
from texar.torch.hyperparams import HParams
from texar.torch.modules import BERTEncoder
from texar.torch.data import BERTTokenizer

from forte.common.resources import Resources
from forte.data import MultiPack
from forte.processors.base import MultiPackProcessor

from forte.data.ontology import Query

__all__ = [
    "QueryCreator"
]


class QueryCreator(MultiPackProcessor):
    r"""This processor is used to search for relevant documents for a query
    """

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, resources: Resources, configs: HParams):
        self.resource = resources
        vocab_file = configs.vocab_file
        self.tokenizer = BERTTokenizer.load(vocab_file)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = BERTEncoder(pretrained_model_name="bert-base-uncased")
        self.encoder.to(self.device)

    @torch.no_grad()
    def get_embeddings(self, input_ids, segment_ids):
        return self.encoder(inputs=input_ids, segment_ids=segment_ids)

    def _process(self, input_pack: MultiPack):
        input_ids = []
        segment_ids = []

        query_pack = input_pack.get_pack("pack")
        context = [query_pack.text]

        # use context to build the query
        if "user_utterance" in input_pack.pack_names:
            user_pack = input_pack.get_pack("user_utterance")
            context.append(user_pack.text)

        if "bot_utterance" in input_pack.pack_names:
            bot_pack = input_pack.get_pack("bot_utterance")
            context.append(bot_pack.text)

        for text in context:
            t = self.tokenizer.encode_text(text)
            input_ids.append(t[0])
            segment_ids.append(t[1])

        input_ids = torch.LongTensor(input_ids).to(self.device)
        segment_ids = torch.LongTensor(segment_ids).to(self.device)
        _, query_vector = self.get_embeddings(input_ids, segment_ids)
        query_vector = torch.mean(query_vector, dim=0, keepdim=True)
        query_vector = query_vector.cpu().numpy()
        query = Query(pack=query_pack, value=query_vector)
        query_pack.add_or_get_entry(query)
