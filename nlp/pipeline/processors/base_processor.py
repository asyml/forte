"""
The base class of processors
"""
from abc import abstractmethod
from typing import Dict, List, Union, Type

from nlp.pipeline.common.resources import Resources
from nlp.pipeline.data import DataPack
from nlp.pipeline.data.ontology import base_ontology, Entry
from nlp.pipeline.utils import get_full_module_name

__all__ = [
    "BaseProcessor",
]


class BaseProcessor:
    """The basic processor class. To be inherited by all kinds of processors
    such as trainer, predictor and evaluator.
    """

    def __init__(self):
        self.component_name = get_full_module_name(self)
        self._ontology = base_ontology
        self._overwrite = True
        self.input_info: Dict[Type[Entry], Union[List, Dict]] = {}
        self.output_info: Dict[Type[Entry], Union[List, Dict]] = {}

    def initialize(self, resource: Resources):
        # TODO Move resource to __init__
        # TODO Change docstring
        """Initialize the processor with ``recources``."""
        pass

    def set_ontology(self, ontology):
        self._ontology = ontology
        self.define_input_info()
        self.define_output_info()

    @abstractmethod
    def define_output_info(self):
        raise NotImplementedError

    @abstractmethod
    def define_input_info(self):
        raise NotImplementedError

    def set_mode(self, overwrite: bool):
        self._overwrite = overwrite

    @abstractmethod
    def process(self, input_pack: DataPack):
        """Process the input data, such as train on the inputs and make
        predictions for the inputs"""
        pass

    def _record_fields(self, data_pack: DataPack):
        """
        Record the fields and entries that this processor add to data packs.
        """
        for entry_type, info in self.output_info.items():
            component = self.component_name
            fields: List[str] = []
            if isinstance(info, list):
                fields = info
            elif isinstance(info, dict):
                fields = info["fields"]
                if "component" in info.keys():
                    component = info["component"]
            data_pack.record_fields(fields, entry_type, component)

    def finish(self, input_pack: DataPack):
        """
        Do finishing work for one data_pack.
        """
        self._record_fields(input_pack)
        input_pack.meta.process_state = self.component_name
