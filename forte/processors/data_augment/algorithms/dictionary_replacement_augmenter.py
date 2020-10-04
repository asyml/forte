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

import nltk
from nltk.corpus import wordnet

from forte.processors.data_augment.algorithms.base_augmenter \
    import ReplacementDataAugmenter



__all__ = [
    "DictionaryReplacementAugmenter",
]

random.seed(0)

class DictionaryReplacementAugmenter(ReplacementDataAugmenter):
    r"""
    This class is a data augmenter utilizing the dictionaries,
    such as WORDNET, to replace the input word with an synonym.
    Part-of-Speech(optional) can be provided to the wordnet for
    retrieving synonyms with the same POS.
    """
    def __init__(self, configs: Dict[str, str]):
        super().__init__(configs)
        try:
            # Check if the wordnet package and
            # pos_tag package are downloaded.
            wordnet.synsets('computer')
            nltk.pos_tag('computer')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
        self.model = wordnet

    @property
    def replacement_level(self) -> List[str]:
        return ["word"]

    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        return WORDNET POS compliance to WORDNET lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return self.model.ADJ
        elif treebank_tag.startswith('V'):
            return self.model.VERB
        elif treebank_tag.startswith('N'):
            return self.model.NOUN
        elif treebank_tag.startswith('R'):
            return self.model.ADV
        else:
            # As default pos in lemmatization is Noun
            return self.model.NOUN


    def augment(self, word: str, pos_tag: str = '') -> str:
        r"""
        This function replaces a word with synonyms from a WORDNET dictionary.
        Args:
            word: input
            additional_info: contains pos_tag of the word, optional
        Returns:
            a synonym of the word
        """
        res: List = []
        pos_wordnet = None
        # The POS property is used for retrieving synonyms with the same POS.
        if len(pos_tag) > 0:
            pos_wordnet = self._get_wordnet_pos(pos_tag)

        for synonym in self.model.synsets(
                word,
                pos=pos_wordnet,
                lang=self.configs['lang']
        ):
            for lemma in synonym.lemmas(lang=self.configs['lang']):
                res.append(lemma.name())
        if len(res) == 0:
            return word
        # Randomly choose one word.
        word = random.choice(res)
        # The phrases are concatenated with "_" in wordnet.
        word = word.replace("_", " ")
        return word
