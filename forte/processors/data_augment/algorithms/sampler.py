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
from typing import Any, Dict, Union
from forte.common.configurable import Configurable

from forte.common.configuration import Config


__all__ = [
    "Sampler",
    "UniformSampler",
    "UnigramSampler",
]


class Sampler(Configurable):
    r"""
    An abstract sampler class.
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]):
        self.configs: Config =  self.make_configs(configs)
        random.seed()

    @abstractmethod
    def sample(self) -> str:
        raise NotImplementedError

    @classmethod
    def default_configs(cls):
        return {}


class UniformSampler(Sampler):
    r"""
    A sampler that samples a word from a uniform distribution.

    Args:
        configs:
            word_list: A list of words that this sampler uniformly samples from.
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]):
        super().__init__(configs)
        # word_list: List[str]
        self.word_list = self.configs["uniform_sampler_word_list"]
        # self.word_list: List[str] = word_list

    def sample(self) -> str:
        word: str = random.choice(self.word_list)
        return word

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update(
            {
                "uniform_sampler_word_list": [],
            }
        )
        return config


class UnigramSampler(Sampler):
    r"""
    A sampler that samples a word from a unigram distribution.

    Config Values:
        - unigram_dict: (dict)
            The key is a word, the value is the word count or a probability.
            This sampler samples from this word distribution.
            Example:

                .. code-block:: python

                    'unigram_dict': {
                        "type": "",
                        "kwargs": {
                            "apple": 1,
                            "banana": 2,
                            "orange": 3
                        }
                    }"""

    def __init__(self, configs: Union[Config, Dict[str, Any]]):
        super().__init__(configs)
        self.unigram = self.configs["unigram_dict"]["kwargs"].__dict__[
            "_hparams"
        ]

    def sample(self) -> str:
        word: str = random.choices(
            list(self.unigram.keys()), list(self.unigram.values())
        )[0]
        return word

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update({"unigram_dict": {"type": "", "kwargs": {}}})
        return config
