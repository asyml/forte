"""
The base class of processors
"""
from abc import abstractmethod
from typing import Dict, List, Union, Type, Generic

from forte.common.resources import Resources
from forte.data import PackType
from forte.data.ontology import base_ontology, Entry
from forte.data import DummySelector
from forte.utils import get_full_module_name

__all__ = [
    "BaseProcessor",
]


class BaseProcessor(Generic[PackType]):
    """The basic processor class. To be inherited by all kinds of processors
    such as trainer, predictor and evaluator.
    """

    def __init__(self):
        self.component_name = get_full_module_name(self)
        self._ontology = base_ontology
        self.input_info: Dict[Type[Entry], Union[List, Dict]] = {}
        self.output_info: Dict[Type[Entry], Union[List, Dict]] = {}
        self.selector = DummySelector()

    def initialize(self, configs, resource: Resources):
        """Initialize the processor with ``configs``, and register global
        resources into ``resource``.
        """
        pass

    @abstractmethod
    def set_ontology(self, ontology):
        """
        Set the ontology of this processor, and accordingly update
        :attr:`input_info`, :attr:`output_info`, and :attr:`context_type` (for
        :class:`~nlp.forte.processors.batch_processor.BatchProcessor`)
        """
        raise NotImplementedError

    @abstractmethod
    def define_output_info(self):
        """
        User should define the output_info here
        """
        raise NotImplementedError

    @abstractmethod
    def define_input_info(self):
        """
        User should define the input_info here
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, input_pack: PackType):
        """Process the input pack"""
        pass

    def _record_fields(self, input_pack: PackType):
        """
        Record the fields and entries that this processor add to packs.
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
            input_pack.record_fields(fields, entry_type, component)

    def finish(self, input_pack: PackType):
        """
        Do finishing work for one pack.
        """
        self._record_fields(input_pack)
        input_pack.meta.process_state = self.component_name

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure
        :return:
        """
        return {
            'selector': {
                'type': 'nlp.forte.data.selector.DummySelector',
                'args': None,
                'kwargs': {}
            }
        }
