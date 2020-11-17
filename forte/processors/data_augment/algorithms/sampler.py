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
from abc import abstractmethod, ABC
from typing import Dict, List, Optional
import nltk
from nltk.corpus import words, stopwords


__all__ = [
    "Sampler",
    "UniformSampler",
    "UnigramSampler",
]


class Sampler(ABC):
    r"""
    A base sampler, an abstract class.
    """
    def __init__(self):
        random.seed()

    @abstractmethod
    def sample(self) -> List[str]:
        raise NotImplementedError


class UniformSampler(Sampler):
    r"""
    Sample from uniform distribution.
    """

    def __init__(self, configs: Optional[Dict]):
        r"""
        A different distribution type can be given in the configs.
        The default distribution is NLTK.
        """
        super().__init__()
        self.configs = configs if configs is not None \
            else self.default_config()

        # set distribution
        if self.configs["distribution"] == "nltk":
            try:
                self.vocab = set(words.words())
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                nltk.download('words')
                nltk.download('stopwords')
                self.vocab = set(words.words('en'))
                self.stop_words = set(stopwords.words('english'))
            self.vocab -= self.stop_words
            self.vocab = list(self.vocab)

        else:
            raise NotImplementedError

    def sample(self) -> str:
        word: str = random.choice(self.vocab)
        return word

    def default_config(self) -> Dict[str, str]:
        return {"distribution": "nltk"}


class UnigramSampler(Sampler):
    r"""
    Sample from a given unigram. A unigram file must be given in configs.
    The file's format:
        word1 count1
        word2 count2
        ......
    """

    def __init__(self, configs: Dict[str, str]):
        super().__init__()
        if "unigram_path" not in configs.keys():
            raise KeyError("unigram file path is missing.")
        self.configs = configs

        self.unigram: Dict[str, int] = self._construct_unigram()

    def _construct_unigram(self) -> Dict[str, int]:
        unigram: Dict[str, int] = {}
        with open(self.configs['unigram_path'], 'r') as file:
            for line in file:
                if len(line) > 0:
                    line = line.strip()
                    word, count = line.split(' ')
                    unigram[word] = int(count)
        return unigram

    def sample(self) -> str:
        word: str = random.choices(list(self.unigram.keys()), list(self.unigram.values()))[0]
        return word
