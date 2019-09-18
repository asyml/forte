"""
The base class of processors
"""
from abc import abstractmethod, ABC
from typing import Dict, List, Union, Type, Generic

from forte.common.resources import Resources
from forte.data import PackType
from forte.data.ontology import base_ontology
from forte.data.selector import DummySelector
from forte.data.data_pack import DataRequest
from forte.utils import get_full_module_name, record_fields

__all__ = [
    "BaseProcessor",
    "ProcessInfo",
]

ProcessInfo = DataRequest


class BaseProcessor(Generic[PackType], ABC):
    """The basic processor class. To be inherited by all kinds of processors
    such as trainer, predictor and evaluator.
    """

    def __init__(self):
        self.component_name = get_full_module_name(self)
        self._ontology = base_ontology
        self.input_info: ProcessInfo = {}
        self.output_info: ProcessInfo = {}
        self.selector = DummySelector()

    def initialize(self, configs, resource: Resources):
        """Initialize the processor with ``configs``, and register global
        resources into ``resource``.
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

    def process(self, input_pack: PackType):
        # TODO finish up the refactors here.
        # Obtain the control of the DataPack.
        input_pack.enter_processing(self.component_name)
        # Do the actual processing.
        self._process(input_pack)
        # Record all the fields.
        record_fields(self.output_info, self.component_name, input_pack)
        # Mark that the pack is processed by the processor.
        input_pack.meta.process_state = self.component_name
        # Release the control of the DataPack.
        input_pack.exit_processing()

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
        raise NotImplementedError

    def finish(self):
        """
        The pipeline will calls this function at the end of the pipeline to
        notify all the processors. The user can implement this function to
        release resources used by this processor.

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
                'type': 'nlp.forte.data.selector.DummySelector',
                'args': None,
                'kwargs': {}
            }
        }
