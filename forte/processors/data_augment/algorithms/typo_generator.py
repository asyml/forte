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
import json


__all__ = [
    "TypoGenerator",
    "UniformTypoGenerator",
]


class TypoGenerator:
    r"""
    An abstract generator class.
    """

    def __init__(self):
        random.seed()

    @abstractmethod
    def generate(self) -> str:
        raise NotImplementedError


class UniformTypoGenerator(TypoGenerator):
    r"""
    A generateor that generates a typo from a typo dictionary.

    Args:
        word_list: A list of words that this sampler uniformly samples from.
    """

    def generate(self, word: str, dict_path: str) -> str:
        with open(dict_path, encoding="utf8") as json_file:
            data = json.load(json_file)
        if word in data.keys():
            result: str = random.choice(data[word])
            return result
        else:
            return word
