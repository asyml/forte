"""
This processor can be used with base_ontology as a blank processor
"""

from nlp.pipeline.processors import PackProcessor
from nlp.pipeline.data import DataPack
from nlp.pipeline.data.ontology import base_ontology

__all__ = [
    "DummyPackProcessor",
]


class DummyPackProcessor(PackProcessor):

    def __init__(self):
        super().__init__()
        self._ontology = base_ontology
        self.define_input_info()
        self.define_output_info()

    def define_input_info(self):
        pass

    def define_output_info(self):
        pass

    def _process(self, input_pack: DataPack):
        pass
