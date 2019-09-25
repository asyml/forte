"""
Defines the Evaluator interface and related functions.
"""
from abc import abstractmethod
from typing import Optional, Generic, Any

from texar.torch import HParams

from forte.common.types import PackType

__all__ = [
    "Evaluator",
]


class Evaluator(Generic[PackType]):
    def __init__(self, config: Optional[HParams] = None):
        self.config: Optional[HParams] = config

    @abstractmethod
    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        """
        Consume the prediction pack and the reference pack to compute evaluation
        results.

        Args:
            pred_pack: The prediction datapack, which should contain the system
            predicted results.
            ref_pack: The reference datapack, which should contain the reference
            to score on.

        Returns:

        """

        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        """
        The Evaluator gather the results and the score can be obtained here.
        Returns:

        """
        raise NotImplementedError
