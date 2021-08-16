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

from typing import Optional

from texar.torch.data.tokenizers.bert_tokenizer import BERTTokenizer

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.utils.utils import DiffAligner
from ft.onto.base_ontology import Subword


# This should probably be named as `BertTokenizer`.
class SubwordTokenizer(PackProcessor):
    """
    Subword Tokenizer using pretrained Bert model.
    """

    def __init__(self):
        self.tokenizer: Optional[BERTTokenizer] = None
        self.aligner: Optional[DiffAligner] = None

    # pylint: disable=attribute-defined-outside-init,unused-argument
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        if not self.configs.tokenizer_configs.pretrained_model_name:
            raise ValueError("Please specify a pretrained bert model")
        self.tokenizer = BERTTokenizer(
            cache_dir=None,
            hparams=self.configs.tokenizer_configs,
        )
        self.aligner = DiffAligner()

    def _process(self, input_pack: DataPack):
        assert self.tokenizer is not None
        assert self.aligner is not None

        # The following logics are adapted from
        # texar.torch.data.tokenzier.BERTTokenizer._map_text_to_token,
        # uses tokenzie_with_span to get the span information. May have
        # problems if there are refactoring happens in BERTTokenizer.
        if self.tokenizer.do_basic_tokenize:
            basic_tokens = self.tokenizer.basic_tokenizer.tokenize(
                input_pack.text, never_split=self.tokenizer.all_special_tokens
            )

            text_to_match = (
                input_pack.text.lower()
                if self.tokenizer.basic_tokenizer.do_lower_case
                else input_pack.text
            )

            token_spans = self.aligner.align_with_segments(
                text_to_match, basic_tokens
            )

            for token, (token_start, token_end) in zip(
                basic_tokens, token_spans
            ):
                assert token is not None

                if token_end <= token_start:
                    # Handle the case where a basic token is not mapped to
                    # the real text span.
                    continue

                for (
                    subword,
                    start,
                    end,
                ) in self.tokenizer.wordpiece_tokenizer.tokenize_with_span(
                    token
                ):
                    subword_token = Subword(
                        input_pack, token_start + start, token_start + end
                    )
                    if subword == self.tokenizer.wordpiece_tokenizer.unk_token:
                        subword_token.is_unk = True
                    subword_token.is_first_segment = not subword.startswith(
                        "##"
                    )
                    # pylint: disable=protected-access
                    subword_token.vocab_id = self.tokenizer._map_token_to_id(
                        subword
                    )
        else:
            for (
                subword,
                start,
                end,
            ) in self.tokenizer.wordpiece_tokenizer.tokenize_with_span(
                input_pack.text
            ):
                subword_token = Subword(input_pack, start, end)
                subword_token.is_first_segment = not subword.startswith("##")

    @classmethod
    def default_configs(cls):
        """Returns the configuration with default values.

        * `tokenizer_configs` contains all default
        hyperparameters in
        :class:`~texar.torch.data.tokenizer.bert_tokenizer.BERTTokenizer`,
        this processor will pass on all the configurations to the tokenizer
        to create the tokenizer instance.

        Returns:

        """
        configs = super().default_configs()
        configs.update({"tokenizer_configs": BERTTokenizer.default_hparams()})

        return configs
