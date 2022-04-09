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
from typing import Tuple, Union, Dict, Any

from forte.common.configuration import Config
from forte.processors.data_augment.algorithms.single_annotation_op import (
    SingleAnnotationAugmentOp,
)
from forte.utils.utils import create_class_with_kwargs

__all__ = [
    "DictionaryReplacementOp",
]

from ft.onto.base_ontology import Token


class DictionaryReplacementOp(SingleAnnotationAugmentOp):
    r"""
    This class is a replacement op utilizing the dictionaries,
    such as WORDNET, to replace the input word with an synonym.
    Part-of-Speech(optional) can be provided to the wordnet for
    retrieving synonyms with the same POS. It will sample from a
    Bernoulli distribution to decide whether to replace the input,
    with `prob` as the probability of replacement.
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]):
        super().__init__(configs)
        self.dictionary = create_class_with_kwargs(
            self.configs["dictionary_class"], class_args={}
        )

    def single_annotation_augment(self, input_anno: Token) -> Tuple[bool, str]:  # type: ignore
        r"""
        This function replaces a word with synonyms from a WORDNET dictionary.

        Args:
            input_anno: The input word.
        Returns:
            A tuple of two values, where the first element is a boolean value
            indicating whether the replacement happens, and the second
            element is the replaced string.
        """
        # If the replacement does not happen, return False.
        if random.random() > self.configs.prob:
            return False, input_anno.text
        word = input_anno.text
        pos_tag = input_anno.pos
        lang = self.configs.lang
        synonyms = self.dictionary.get_synonyms(word, pos_tag, lang)
        if len(synonyms) == 0:
            return False, word
        return True, random.choice(synonyms)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        r"""
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - `dictionary` (dict):
                The full qualified name of the dictionary class.
            - `prob` (float):
                The probability of replacement, should fall in [0, 1].
                Default value is 0.1
            - `lang` (str):
                The language of the text.
        """

        dict_name = (
            "forte.processors.data_augment."
            "algorithms.dictionary.WordnetDictionary"
        )
        return {
            "dictionary_class": dict_name,
            "prob": 0.5,
            "lang": "eng",
        }
