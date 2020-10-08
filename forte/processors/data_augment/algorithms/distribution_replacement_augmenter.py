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
from nltk.corpus import words, stopwords

from forte.processors.data_augment.algorithms.base_augmenter \
    import ReplacementDataAugmenter


__all__ = [
    "DistributionReplacementAugmenter",
]


class DistributionReplacementAugmenter(ReplacementDataAugmenter):
    r"""
    This class is a data augmenter utilizing the NLTK vocabulary,
    from which sample by a given distribution to subsitute a word
    from the original text. The distributions implemented now are uniform,
    unigram (related to a specific dataset).
    Todo: More distriubtion types can be added.
    """
    def __init__(self, configs: Dict[str, str]):
        random.seed(0)
        super().__init__(configs)
        try:
            self.vocab = set(words.words())
            self.stop_words = set(stopwords.words(self.configs['lang']))
        except LookupError:
            nltk.download('words')
            nltk.download('stopwords')
            self.vocab = set(words.words('en'))
            self.stop_words = set(stopwords.words(self.configs['lang']))

    @property
    def replacement_level(self) -> List[str]:
        return ["word"]

    def _uniform_sample(self) -> str:
        """
        return a random word from vocab
        """
        word: str = random.choice(self.vocab - self.stop_words)[0]
        return word

    def _unigram_sample(self) -> str:
        """
        Load the unigram distribution of one dataset from file and
        return a word sampled from unigram distribution
        """
        vocab, cnt = [], []
        with open(self.configs['unigram'], 'r') as file:
            for line in file:
                if len(line) > 0:
                    line = line.strip()
                    word, count = line.split(' ')
                    vocab.append(word)
                    cnt.append(int(count))
        return random.choices(vocab, cnt)[0]

    def augment(self) -> str:
        r"""
        This function replaces a word from sampling a distribution.
        Args:
        Returns:
            a word sampled from distribution
        """
        if self.configs['distribution'] == "uniform":
            return self._uniform_sample()
        elif self.configs['distribution'] == "unigram":
            if self.configs['unigram'] == "":
                raise KeyError("Unigram distribution is not "
                               "provided in config.")
            else:
                return self._unigram_sample()
        else:
            raise NotImplementedError

    @classmethod
    def default_configs(cls):
        """
        :return: A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - distribution: defines what types of distribution to be used.
                Options are uniform, unigram, etc.
            - unigram: An input file with unigram stats of the specific dataset.
                if distribution type is set to "unigram" and "unigram"
                file is not provided,
                then simply use _generate_unigram function to generate one
                from the given text.
        """
        config = super().default_configs()
        config.update({
            'distribution': "uniform",
            'unigram': ""
        })
        return config
