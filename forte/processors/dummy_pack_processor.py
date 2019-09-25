"""
This processor can be used with base_ontology as a blank processor
"""

from forte.data import DataPack
from forte.data.ontology import base_ontology
from forte.processors.base import PackProcessor, ProcessInfo

__all__ = [
    "DummyPackProcessor",
]


class DummyPackProcessor(PackProcessor):

    def __init__(self):
        super().__init__()
        self._ontology = base_ontology

    def _define_input_info(self) -> ProcessInfo:
        pass

    def _define_output_info(self) -> ProcessInfo:
        pass

    def _process(self, input_pack: DataPack):
        pass
