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
import pickle
from typing import Any, Dict, Tuple

import numpy as np

try:
    import torch
except ImportError as e1:
    raise ImportError(
        " `pytorch` is not installed correctly."
        " Consider install torch "
        "via `pip install torch`."
        " Or refer to [extra requirement for ir processors](pip install forte[ir])"
        " for more information. "
    ) from e1


from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base import QueryProcessor

__all__ = ["BertBasedQueryCreator"]


class BertBasedQueryCreator(QueryProcessor):
    r"""This processor searches relevant documents for a query"""

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, resources: Resources, configs: Config):
        self.resource = resources
        self.config = configs

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            from texar.torch.data import (  # pylint: disable=import-outside-toplevel
                BERTTokenizer,
            )
            from texar.torch.modules import (  # pylint: disable=import-outside-toplevel
                BERTEncoder,
            )
        except ImportError as e:
            raise ImportError(
                " `texar-pytorch` is not installed correctly."
                " Consider install texar via `pip install texar-pytorch`"
                " Or refer to [extra requirement for IR support](pip install forte[ir])"
                " for more information."
            ) from e

        if "name" in self.config.tokenizer:
            self.tokenizer = BERTTokenizer(
                pretrained_model_name=self.config.tokenizer.name
            )

        if "name" in self.config.model:
            self.encoder = BERTEncoder(
                pretrained_model_name=self.config.model.name
            )

        else:
            self.encoder = BERTEncoder(
                pretrained_model_name=None,
                hparams={"pretrained_model_name": None},
            )
            with open(self.config.model.path, "rb") as f:
                state_dict = pickle.load(f)
            self.encoder.load_state_dict(state_dict["bert"])

        self.encoder.to(self.device)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "model": {
                "path": None,
                "name": "bert-base-uncased",
            },
            "tokenizer": {"name": "bert-base-uncased"},
            "max_seq_length": 128,
            "query_pack_name": "query",
        }

    @torch.no_grad()
    def get_embeddings(self, inputs, sequence_length, segment_ids):
        output, _ = self.encoder(
            inputs=inputs,
            sequence_length=sequence_length,
            segment_ids=segment_ids,
        )
        cls_token = output[:, 0, :]

        return cls_token

    def _build_query(self, text: str) -> np.ndarray:
        (input_ids, segment_ids, input_mask,) = self.tokenizer.encode_text(
            text_a=text, max_seq_length=self.config.max_seq_length
        )
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        segment_ids = torch.LongTensor(segment_ids).unsqueeze(0).to(self.device)
        input_mask = torch.LongTensor(input_mask).unsqueeze(0).to(self.device)
        sequence_length = (~(input_mask == 0)).sum(dim=1)
        query_vector = self.get_embeddings(
            inputs=input_ids,
            sequence_length=sequence_length,
            segment_ids=segment_ids,
        )
        query_vector = torch.mean(query_vector, dim=0, keepdim=True)
        query_vector = query_vector.cpu().numpy()
        return query_vector

    def _process_query(
        self, input_pack: MultiPack
    ) -> Tuple[DataPack, np.ndarray]:
        query_pack: DataPack = input_pack.get_pack(self.config.query_pack_name)
        context = [query_pack.text]

        # use context to build the query
        if "user_utterance" in input_pack.pack_names:
            user_pack = input_pack.get_pack("user_utterance")
            context.append(user_pack.text)

        if "bot_utterance" in input_pack.pack_names:
            bot_pack = input_pack.get_pack("bot_utterance")
            context.append(bot_pack.text)

        text = " ".join(context)

        query_vector = self._build_query(text=text)

        return query_pack, query_vector
