from abc import abstractmethod
from typing import Dict, Any, List
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.utils import *
from nlp.pipeline.common.resources import Resources

__all__ = [
    "BaseProcessor",
]


class BaseProcessor:
    """The basic processor class. To be inherited by all kinds of processors
    such as trainer, predictor and evaluator.
    """
    def __init__(self):
        self.component_name = get_full_component_name(self)
        self.context_type = None
        self.annotation_types = None
        self.link_types = None
        self.group_types = None
        self.batch_size = None

    @abstractmethod
    def initialize(self, resource: Resources):
        """Initialize the processor with ``recources``."""
        pass

    @abstractmethod
    def process(self, *inputs):
        """Process the input data, such as train on the inputs and make
        predictions for the inputs"""
        pass

    @abstractmethod
    def finish(self):
        """Do Clean up work such as releasing the model."""
        pass
