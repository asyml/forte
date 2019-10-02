"""
This file contains examples of PackProcessor implementations, the processors
here are useful as placeholders and test cases.
"""

from forte.data import DataPack
from forte.processors.base import PackProcessor, ProcessInfo

__all__ = [
    "DummyPackProcessor",
]


class DummyPackProcessor(PackProcessor):
    def __init__(self):  # pylint: disable=useless-super-delegation
        super().__init__()

    def _define_input_info(self) -> ProcessInfo:
        pass

    def _define_output_info(self) -> ProcessInfo:
        pass

    def _process(self, input_pack: DataPack):
        pass
