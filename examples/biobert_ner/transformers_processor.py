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

from typing import Dict, Any

from nltk.tokenize.util import align_tokens
from transformers import AutoTokenizer

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Subword


class BERTTokenizer(PackProcessor):
    r"""A wrapper of BERT tokenizer.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None

    # pylint: disable=unused-argument
    def initialize(self, resources: Resources, configs: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(configs.model_path)

    def _process(self, input_pack: DataPack):
        inputs = self.tokenizer(input_pack.text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(
                     inputs['input_ids'][0].tolist()
                 )[1:-1]
        tokens_clean = [token.replace('##', '') if token.startswith('##')
                        else token for token in tokens]

        for i, (begin, end) in enumerate(align_tokens(tokens_clean,
                                                      input_pack.text.lower())):
            subword = Subword(input_pack, begin, end)
            subword.is_first_segment = not tokens[i].startswith('##')

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        r"""Returns a `dict` of configurations of the processor with default
        values. Used to replace the missing values of input ``configs`` during
        pipeline construction.
        """
        config = super().default_configs()
        config.update({'model_path': None})
        return config
