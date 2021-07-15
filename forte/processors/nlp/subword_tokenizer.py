# Copyright 2021 The Forte Authors. All Rights Reserved.
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
"""Subword Tokenizer"""

__all__ = [
    "SubwordTokenizer",
]

from texar.torch.data.tokenizers.bert_tokenizer import BERTTokenizer

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Subword


class SubwordTokenizer(PackProcessor):
    """
    Subword Tokenizer using pretrained Bert model.
    """

    # pylint: disable=attribute-defined-outside-init,unused-argument
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        self.resources = resources
        self.config = Config(configs, self.default_configs())
        if not self.config.pretrained_model_name:
            raise ValueError("Please specify a pretrained bert model")
        self.tokenizer = BERTTokenizer(
            pretrained_model_name=self.config.pretrained_model_name,
            cache_dir=None,
            hparams=None,
        )

    def _process(self, input_pack: DataPack):
        subword_tokenizer = self.tokenizer.wordpiece_tokenizer
        subwords = subword_tokenizer.tokenize_with_span(input_pack.text)
        for subword, start, end in subwords:
            subword_token = Subword(input_pack, start, end)
            subword_token.is_first_segment = not subword.startswith("##")

    @classmethod
    def default_configs(cls):
        configs = super().default_configs()
        configs.update(
            {
                "pretrained_model_name": "bert-base-uncased",
            }
        )
        return configs
