# Copyright 2020 The Forte Authors. All Rights Reserved.
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

import random
from typing import Dict, List
from forte.processors.data_augment.algorithms.base_augmenter import ReplacementDataAugmenter
from forte.processors.base import MultiPackProcessor
from forte.data.multi_pack import MultiPack


__all__ = [
    "BackTranslationAugmenter",
]

random.seed(0)

class BackTranslationAugmenter(ReplacementDataAugmenter):
    r"""
    This class is a data augmenter using back translation to generate data with
    the same semantic meanings. The input is translated to another language, then
    translated back to the original language, with pretrained machine-translation
    models.
    """
    def __init__(self, model_to: MultiPackProcessor, model_back: MultiPackProcessor, configs: Dict[str, str]):
        super().__init__(configs)
        self.model_to = model_to
        self.model_back = model_back

        configs_model_to = {
            "src_language": configs.src_language,
            "tgt_language": configs.tgt_language,
            "in_pack_name": "bt_input_pack",
            "out_pack_name": "bt_tmp_pack"
        }

        configs_model_back = {
            "src_language": configs.tgt_language,
            "tgt_language": configs.src_language,
            "in_pack_name": "bt_tmp_pack",
            "out_pack_name": "bt_output_pack"
        }

        self.model_to.initialize(configs=configs_model_to)
        self.model_back.initialize(configs=configs_model_back)


    @property
    def replacement_level(self) -> List[str]:
        return ["word", "sentence", "Document"]



    def augment(self, input: str) -> str:
        r"""
        This function replaces a word with synonyms from a WORDNET dictionary.
        Args:
            input: a string, could be a word, sentence or document.
        Returns:
            a string with a similar semantic meaning.
        """
        multi_pack: MultiPack = MultiPack()
        multi_pack.add_pack("bt_input_pack")
        input_pack: DataPack = multi_pack.get_pack("bt_input_pack")
        input_pack.set_text(input)
        self.model_to._process(multi_pack)
        self.model_back._process(multi_pack)
        return multi_pack.get_pack("bt_output_pack").text


