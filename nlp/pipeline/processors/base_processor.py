"""
The base class of processors
"""
from abc import abstractmethod
from nlp.pipeline.utils import *
from nlp.pipeline.common.resources import Resources
from nlp.pipeline.data import DataPack
from nlp.pipeline.data.ontology import base_ontology

__all__ = [
    "BaseProcessor",
]


class BaseProcessor:
    """The basic processor class. To be inherited by all kinds of processors
    such as trainer, predictor and evaluator.
    """

    def __init__(self):
        self.component_name = get_full_component_name(self)
        self.ontology = base_ontology

    def initialize(self, resource: Resources):
        # TODO Move resource to __init__
        # TODO Change docstring
        """Initialize the processor with ``recources``."""
        pass

    @abstractmethod
    def process(self, input_pack: DataPack):
        """Process the input data, such as train on the inputs and make
        predictions for the inputs"""
        pass

    @abstractmethod
    def _record_fields(self, data_pack: DataPack):
        """
        Record the fields and entries that this processor add to data packs.
        """
        pass

    def finish(self, input_pack: DataPack = None):
        """
        Do finishing work for one data_pack.
        """
        self._record_fields(input_pack)
        input_pack.meta.process_state = self.component_name
        # currently, need to build the coverage index after updating the entries
        input_pack.index.build_coverage_index(
            input_pack.annotations,
            input_pack.links,
            input_pack.groups,
            outer_type=self.ontology.Sentence
        )