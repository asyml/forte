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

from forte.data.ontology import Annotation
from forte.utils.utils import create_class_with_kwargs
from forte.common.configuration import Config
from forte.processors.data_augment.algorithms.text_replacement_op import (
    TextReplacementOp,
)

__all__ = [
    "DictionaryReplacementOp",
]


class DictionaryReplacementOp(TextReplacementOp):
    r"""
    This class is a replacement op utilizing the dictionaries,
    such as WORDNET, to replace the input word with an synonym.
    Part-of-Speech(optional) can be provided to the wordnet for
    retrieving synonyms with the same POS. It will sample from a
    Bernoulli distribution to decide whether to replace the input,
    with `prob` as the probability of replacement.

    The config should contain the following fields:
        - `dictionary`: The full qualified name of the dictionary class.
        - `prob`: The probability of replacement, should fall in [0, 1].
        - `lang`: The language of the text.
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]):
        super().__init__(configs)
        self.dictionary = create_class_with_kwargs(
            configs["dictionary_class"], class_args={}
        )

    def replace(self, input_anno: Annotation) -> Tuple[bool, str]:
        r"""
        This function replaces a word with synonyms from a WORDNET dictionary.

        Args:
            input_anno (Annotation): The input annotation.
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
