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
"""
This file implements SubwordExtractor, which is used to extract feature
from the subwords of an entry.
"""
import logging
from typing import Union, Dict, Optional

from texar.torch.data.tokenizers.bert_tokenizer import BERTTokenizer
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.converter.feature import Feature
from forte.data.base_extractor import BaseExtractor
from forte.data.ontology import Annotation

logger = logging.getLogger(__name__)

__all__ = ["SubwordExtractor"]


class SubwordExtractor(BaseExtractor):
    r"""SubwordExtractor extracts feature from the subword of entry.
    Most of the time, a user will not need to call this class explicitly,
    they will be called by the framework.

    Args:
        config: An instance of `Dict`  :class:`~forte.common.configuration.Config`
    """

    def initialize(self, config: Union[Dict, Config]):
        # pylint: disable=attribute-defined-outside-init
        super().initialize(config=config)
        self.tokenizer = BERTTokenizer(
            pretrained_model_name=self.config.pretrained_model_name,
            cache_dir=None,
            hparams=None,
        )
        predefined_dict = [key for key, _ in self.tokenizer.vocab.items()]
        self.predefined_vocab(predefined_dict)
        if not self.vocab:
            raise AttributeError("Vocabulary is required in SubwordExtractor.")
        self.vocab.mark_special_element(self.tokenizer.vocab["[PAD]"], "PAD")
        self.vocab.mark_special_element(self.tokenizer.vocab["[UNK]"], "UNK")

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        Here:

        - "`pretrained_model_name`":
            The name of the pretrained bert model. Must be the same
            as used in subword tokenizer.
        - "`subword_class`": the fully qualified name of the class of the
            subword, default is `ft.onto.base_ontology.Subword`.
        """
        config = super().default_configs()
        config.update(
            {
                "pretrained_model_name": None,
                "subword_class": "ft.onto.base_ontology.Subword",
            }
        )
        return config

    def extract(
        self, pack: DataPack, context: Optional[Annotation] = None
    ) -> Feature:
        r"""Extract the subword feature of one instance.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            context (Annotation): The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.

        Returns:
            Feature: a feature that contains the extracted data.
        """
        data = []

        subword: Annotation
        for subword in pack.get(self.config.subword_class, context):
            text = subword.text  # type: ignore
            if not subword.is_first_segment:  # type: ignore
                text = "##" + text
            data.append(self.element2repr(text))

        data = (
            [self.element2repr("[CLS]")] + data + [self.element2repr("[SEP]")]
        )

        meta_data = {
            "need_pad": self.vocab.use_pad,  # type: ignore
            "pad_value": self.get_pad_value(),
            "dim": 1,
            "dtype": int,
        }

        return Feature(data=data, metadata=meta_data, vocab=self.vocab)
