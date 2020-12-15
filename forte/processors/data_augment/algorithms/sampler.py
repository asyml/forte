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
from typing import Dict, List


__all__ = [
    "Sampler",
    "UniformSampler",
    "UnigramSampler",
]


class Sampler(ABC):
    r"""
    An abstract sampler class.
    """
    def __init__(self):
        random.seed()

    @abstractmethod
    def sample(self) -> str:
        raise NotImplementedError


class UniformSampler(Sampler):
    r"""
    A sampler that samples a word from a uniform distribution.

    Args:
        word_list: A list of words that this sampler uniformly samples from.
    """

    def __init__(self, word_list: List[str]):
        super().__init__()
        self.word_list: List[str] = word_list

    def sample(self) -> str:
        word: str = random.choice(self.word_list)
        return word


class UnigramSampler(Sampler):
    r"""
    A sampler that samples a word from a unigram distribution.

    Args:
        unigram: A dictionary.
            The key is a word, the value is the word count or a probability.
            This sampler samples from this word distribution.
    """

    def __init__(self, unigram: Dict[str, float]):
        super().__init__()
        self.unigram: Dict[str, float] = unigram

    def sample(self) -> str:
        word: str = random.choices(list(self.unigram.keys()),
                                   list(self.unigram.values()))[0]
        return word
