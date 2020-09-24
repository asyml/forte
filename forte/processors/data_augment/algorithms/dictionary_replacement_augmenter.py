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
from forte.processors.data_augment.algorithms.base_augmenter import BaseDataAugmenter

__all__ = [
    "DictionaryReplaceAugmenter",
]

random.seed(0)

class DictionaryReplacementAugmenter(BaseDataAugmenter):
    r"""
    This class is a data augmenter utilizing the dictionaries,
    such as WORDNET, to replace the input word with an synonym.
    Part-of-Speech(optional) can be provided to the wordnet for
    retrieving synonyms with the same POS.
    """
    def __init__(self, configs: Dict[str, str]):
        self.configs = configs
        # check if the nltk is properly installed
        try:
            import nltk
            from nltk.corpus import wordnet
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Missed nltk library. Please install it by 'pip install nltk'")

        try:
            # Check if the wordnet package and pos_tag package are downloaded.
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


    def augment(self, word: str, additional_info: Dict[str, str] = {}) -> str:
        r"""
        This function replaces a word with synonyms from a WORDNET dictionary.
        :param word: input
        :param additional_info: contains pos_tag of the word, optional
        :return: a synonym of the word
        """
        res: List = []
        pos = None
        # The POS property is used for retrieving synonyms with the same POS.
        if 'pos_tag' in additional_info:
            pos = self._get_wordnet_pos(additional_info['pos_tag'])

        for synonym in self.model.synsets(word, pos=pos, lang=self.configs['lang']):
            for lemma in synonym.lemmas(lang=self.configs['lang']):
                res.append(lemma.name())
        if len(res) == 0:
            return word
        # Randomly choose one word.
        word = random.choice(res)
        # The phrases are concatenated with "_" in wordnet.
        word = word.replace("_", " ")
        return word
