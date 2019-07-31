"""
Defines the Evaluator interface and related functions.
"""
from abc import abstractmethod

from nlp.pipeline.data.data_pack import DataPack

__all__ = [
    "Evaluator",
]


class Evaluator:
    def __init__(self, config=None):
        pass

    @abstractmethod
    def consume_next(self, pred_pack: DataPack, ref_pack: DataPack):
        # TODO: We may want to adjust this function signature.
        # I had used (self, *args, **kwargs)
        # But in the NEREvaluator we have to override the function signature
        raise NotImplementedError

    @abstractmethod
    def get_result(self):
        raise NotImplementedError
