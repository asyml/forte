"""
The base class of processors
"""
from abc import abstractmethod, ABC
from typing import Dict, List, Union, Type, Generic

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data import PackType
from forte.data.ontology import base_ontology, Entry
from forte.data.selector import DummySelector
from forte.utils import get_full_module_name

__all__ = [
    "BaseProcessor",
    "ProcessInfo",
]

ProcessInfo = Dict[Type[Entry], Union[List, Dict]]


class BaseProcessor(Generic[PackType], ABC):
    """
    The basic processor class. To be inherited by all kinds of processors
    such as trainer, predictor and evaluator.
    """

    def __init__(self):
        self.component_name = get_full_module_name(self)
        self._ontology = base_ontology
        self.input_info: ProcessInfo = {}
        self.output_info: ProcessInfo = {}
        self.selector = DummySelector()

    def initialize(self, configs: HParams, resource: Resources):
        """
        The pipeline will call the initialize method at the start of a
        processing. The processor will be initialized with ``configs``,
        and register global resources into ``resource``.

        :param configs: The configuration passed in to set up this processor.
        :param resource: A global resource register. User can register
        shareable resources here, for example, the vocabulary.
        :return:
        """
        pass

    def set_ontology(self, ontology):
        """
        Set the ontology of this processor, will be called by the Pipeline.
        """
        self._ontology = ontology  # pylint: disable=attribute-defined-outside-init

    def set_output_info(self):
        self.output_info = self._define_output_info()

    def set_input_info(self):
        self.input_info = self._define_input_info()

    @abstractmethod
    def _define_input_info(self) -> ProcessInfo:
        """
        User should define the input_info here
        """
        raise NotImplementedError

    @abstractmethod
    def _define_output_info(self) -> ProcessInfo:
        """
        User should define the output_info here
        """
        raise NotImplementedError

    # TODO: understand tail_instances
    def process_internal(self, input_pack: PackType):
        # TODO finish up the refactors here.
        self._process(input_pack)
        self._record_fields(input_pack)
        input_pack.meta.process_state = self.component_name

    @abstractmethod
    def _process(self, input_pack: PackType):
        """
        The main function of the processor should be implemented here. The
        implementation of this function should process the ``input_pack``, and
        conduct operations such as adding entries into the pack, or produce
        some side-effect such as writing data into the disk.

        Args:
            input_pack:

        Returns:

        """
        """Process the input pack"""
        raise NotImplementedError

    def _record_fields(self, input_pack: PackType):
        """
        Record the fields and entries that this processor add to packs.

        Args:
            input_pack:

        Returns:
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

    def finish(self):
        """
        The user can implement this function to close and release resources
        used by this processor.

        Returns:

        """
        pass

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure
        :return:
        """
        return {
            'selector': {
                'type': 'forte.data.selector.DummySelector',
                'args': None,
                'kwargs': {}
            }
        }
