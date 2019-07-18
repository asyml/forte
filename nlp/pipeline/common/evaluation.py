"""
Defines the Evaluator interface and related functions.
"""
from abc import abstractmethod
from nlp.pipeline.data import DataPack

__all__ = [
    "Evaluator",
]


class Evaluator:
    def __init__(self, config):
        pass

    @abstractmethod
    def consume_next(self, pack: DataPack):
        raise NotImplementedError

    @abstractmethod
    def get_result(self):
        raise NotImplementedError
