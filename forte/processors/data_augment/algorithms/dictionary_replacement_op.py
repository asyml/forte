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
from ft.onto.base_ontology import Entry
from forte.utils.utils import create_class_with_kwargs
from forte.common.configuration import Config
from forte.processors.data_augment.algorithms.text_replacement_op \
    import TextReplacementOp

__all__ = [
    "DictionaryReplacementOp",
]


class DictionaryReplacementOp(TextReplacementOp):
    r"""
    This class is a replacement op utilizing the dictionaries,
    such as WORDNET, to replace the input word with an synonym.
    Part-of-Speech(optional) can be provided to the wordnet for
    retrieving synonyms with the same POS.

    The config should contain the following fields:
        - dictionary: The full path to the dictionary class.
        - lang: The language of the text.
    """
    def __init__(self, configs: Config):
        super().__init__(configs)
        self.dictionary = create_class_with_kwargs(
            configs["dictionary"],
            class_args={}
        )

    def replace(self, token: Entry) -> str:
        r"""
        This function replaces a word with synonyms from a WORDNET dictionary.
        Args:
            token: The input entry.
        Returns:
            A synonym of the word.
        """
        word = token.text
        pos_tag = token.pos
        lang = self.configs["lang"]
        return self.dictionary.get_synonyms(word, pos_tag, lang)
