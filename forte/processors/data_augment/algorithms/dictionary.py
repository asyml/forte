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
from abc import abstractmethod
from typing import List
import nltk
from nltk.corpus import wordnet


__all__ = [
    "Dictionary",
    "WordnetDictionary"
]


class Dictionary:
    r"""
    This class defines a dictionary for word replacement.
    Given an input word and its pos_tag(optional), the dictionary
    will outputs its synonyms.
    """
    @abstractmethod
    def get_synonyms(self, word, pos_tag: str = "", lang: str = "eng"):
        r"""
        Args:
            - word: The input string.
            - pos_tag: The Part-of-Speech tag for substitution.
            - lang: The language of the input string.
        Returns:
            A synonym of the word.
        """
        raise NotImplementedError


class WordnetDictionary(Dictionary):
    r"""
    This class wraps the nltk WORDNET to replace
    the input word with an synonym. Part-of-Speech(optional)
    can be provided to the wordnet for retrieving
    synonyms with the same POS.
    """
    def __init__(self):
        try:
            # Check if the wordnet package and
            # pos_tag package are downloaded.
            wordnet.synsets('computer')
        except LookupError:
            nltk.download('wordnet')
        self.model = wordnet

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

    def get_synonyms(self, word, pos_tag: str = "", lang: str = "eng"):
        r"""
        This function replaces a word with synonyms from a WORDNET dictionary.
        """
        res: List[str] = []
        pos_wordnet = None
        # The POS property is used for retrieving synonyms with the same POS.
        if pos_tag and len(pos_tag) > 0:
            pos_wordnet = self._get_wordnet_pos(pos_tag)

        for synonym in self.model.synsets(
            word,
            pos=pos_wordnet,
            lang=lang
        ):
            for lemma in synonym.lemmas(lang=lang):
                res.append(lemma.name())
        if len(res) == 0:
            return word
        # Randomly choose one word.
        word = random.choice(res)
        # The phrases are concatenated with "_" in wordnet.
        word = word.replace("_", " ")
        return word
