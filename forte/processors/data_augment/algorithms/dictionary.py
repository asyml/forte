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
    will outputs its synonyms, antonyms, hypernyns and hyponyms.
    """

    # pylint: disable=unused-argument
    def get_synonyms(
            self,
            word: str,
            pos_tag: str = "",
            lang: str = "eng"
    ) -> List[str]:
        r"""
        Args:
            word (str): The input string.
            pos_tag (str): The Part-of-Speech tag for substitution.
            lang (str): The language of the input string.
        Returns:
            synonyms of the word.
        """
        return []

    def get_antonyms(
            self,
            word: str,
            pos_tag: str = "",
            lang: str = "eng"
    ) -> List[str]:
        r"""
        Args:
            word (str): The input string.
            pos_tag (str): The Part-of-Speech tag for substitution.
            lang (str): The language of the input string.
        Returns:
            Antonyms of the word.
        """
        return []

    def get_hypernyms(
            self,
            word: str,
            pos_tag: str = "",
            lang: str = "eng"
    ) -> List[str]:
        r"""
        Args:
            word (str): The input string.
            pos_tag (str): The Part-of-Speech tag for substitution.
            lang (str): The language of the input string.
        Returns:
            Hypernyms of the word.
        """
        return []

    def get_hyponyms(
            self,
            word: str,
            pos_tag: str = "",
            lang: str = "eng"
    ) -> List[str]:
        r"""
        Args:
            word (str): The input string.
            pos_tag (str): The Part-of-Speech tag for substitution.
            lang (str): The language of the input string.
        Returns:
            Hyponyms of the word.
        """
        return []


class WordnetDictionary(Dictionary):
    r"""
    This class wraps the nltk WORDNET to replace
    the input word with an synonym/antonym/hypernym/hyponym.
    Part-of-Speech(optional) can be provided to the wordnet
    for retrieving words with the same POS.
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

    def get_lemmas(
            self,
            word: str,
            pos_tag: str = "",
            lang: str = "eng",
            lemma_type: str = "SYNONYM"
    ):
        r"""
        This function gets synonyms/antonyms/hypernyms/hyponyms
        from a WORDNET dictionary.

        Args:
            word (str): The input token.
            pos_tag (str): The NLTK POS tag.
            lang (str): The input language.
            lemma_type (str): The type of words to replace, must be
                one of the following:

                - ``'SYNONYM'``
                - ``'ANTONYM'``
                - ``'HYPERNYM'``
                - ``'HYPONYM'``
        """
        res: List[str] = []
        pos_wordnet = None
        # The POS property is used for retrieving lemmas with the same POS.
        if pos_tag and len(pos_tag) > 0:
            pos_wordnet = self._get_wordnet_pos(pos_tag)

        for synonym in self.model.synsets(
            word,
            pos=pos_wordnet,
            lang=lang
        ):
            for lemma in synonym.lemmas(lang=lang):
                if lemma_type == "SYNONYM":
                    res.append(lemma.name())
                elif lemma_type == "ANTONYM":
                    for antonym in lemma.antonyms():
                        res.append(antonym.name())
                elif lemma_type == "HYPERNYM":
                    for hypernym in lemma.hypernyms():
                        res.append(hypernym.name())
                elif lemma_type == "HYPONYM":
                    for hyponym in lemma.hyponyms():
                        res.append(hyponym.name())
                else:
                    raise KeyError(
                        'The type {} does not belong to '
                        '["SYNONYM", "ANTONYM", '
                        '"HYPERNYM", "HYPONYM"]]'.format(type)
                    )
        # The phrases are concatenated with "_" in wordnet.
        return [word.replace("_", " ") for word in res]

    def get_synonyms(
            self,
            word: str,
            pos_tag: str = "",
            lang: str = "eng"
    ) -> List[str]:
        r"""
        This function replaces a word with synonyms from a WORDNET dictionary.
        """
        return self.get_lemmas(word, pos_tag, lang, lemma_type="SYNONYM")

    def get_antonyms(
            self,
            word: str,
            pos_tag: str = "",
            lang: str = "eng"
    ) -> List[str]:
        r"""
        This function replaces a word with antonyms from a WORDNET dictionary.
        """
        return self.get_lemmas(word, pos_tag, lang, lemma_type="ANTONYM")

    def get_hypernyms(
            self,
            word: str,
            pos_tag: str = "",
            lang: str = "eng"
    ) -> List[str]:
        r"""
        This function replaces a word with hypernyms from a WORDNET dictionary.
        """
        return self.get_lemmas(word, pos_tag, lang, lemma_type="HYPERNYM")

    def get_hyponyms(
            self,
            word: str,
            pos_tag: str = "",
            lang: str = "eng"
    ) -> List[str]:
        r"""
        This function replaces a word with hyponyms from a WORDNET dictionary.
        """
        return self.get_lemmas(word, pos_tag, lang, lemma_type="HYPONYM")
