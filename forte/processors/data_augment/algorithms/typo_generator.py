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
