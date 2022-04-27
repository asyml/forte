# Copyright 2022 The Forte Authors. All Rights Reserved.
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
import json
from typing import Tuple, Dict, Any

import requests
from ft.onto.base_ontology import Phrase
from forte.processors.data_augment.algorithms.single_annotation_op import (
    SingleAnnotationAugmentOp,
)
from forte.common.configuration import Config

__all__ = [
    "AbbreviationReplacementOp",
]


class AbbreviationReplacementOp(SingleAnnotationAugmentOp):
    r"""
    This class is a replacement op utilizing a pre-defined
    abbreviation to replace words, to replace the input phrase
    with an abbreviation.

    Args:
        configs:
            - prob: The probability of replacement,
              should fall in [0, 1].
            - dict_path: the `url` or the path to the pre-defined
             abbreviation json file. The key is a word / phrase we want to replace.
             The value is an abbreviated word of the corresponding key.
    """

    def __init__(self, configs: Config):
        super().__init__(configs)

        dict_path = configs["dict_path"]

        try:
            r = requests.get(dict_path)
            self.data = r.json()
        except requests.exceptions.RequestException:
            with open(dict_path, encoding="utf8") as json_file:
                self.data = json.load(json_file)

    def single_annotation_augment(
        self, input_phrase: Phrase
    ) -> Tuple[bool, str]:
        r"""
        This function replaces a phrase from an abbreviation dictionary
        with `prob` as the probability of replacement.
        If the input phrase does not have a corresponding phrase in the
        dictionary, no replacement will happen, return False.

        Args:
            input_phrase: The input phrase.
        Returns:
            A tuple, where the first element is a boolean value indicating
            whether the replacement happens, and the second element is the
            replaced string.
        """
        # If the replacement does not happen, return False.
        if random.random() > self.configs.prob:
            return False, input_phrase.text
        if input_phrase.text in self.data.keys():
            result: str = self.data[input_phrase.text]
            return True, result
        else:
            return False, input_phrase.text

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        r"""
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - prob: The probability of replacement,
              should fall in [0, 1]. Default value is 0.5.
            - dict_path: the `url` or the path to the pre-defined
              abbreviation json file. The key is a word / phrase we want
              to replace. The value is an abbreviated word of the
              corresponding key. Default dictionary is from a web-scraped
              slang dictionary ("https://github.com/abbeyyyy/JsonFiles/
              blob/main/abbreviate.json").
        """
        return {
            "augment_entry": "ft.onto.base_ontology.Phrase",
            "other_entry_policy": {
                "ft.onto.base_ontology.Phrase": "auto_align",
            },
            "dict_path": "https://raw.githubusercontent.com/abbeyyyy/"
            "JsonFiles/main/abbreviate.json",
            "prob": 0.5,
        }
