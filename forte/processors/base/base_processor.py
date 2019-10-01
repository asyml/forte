"""
The base class of processors
"""
from abc import abstractmethod, ABC
from typing import Optional

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.data.data_pack import DataRequest
from forte.data.ontology import base_ontology
from forte.data.selector import DummySelector
from forte.utils import get_full_module_name
from forte.data.ontology.onto_utils import record_fields
from forte.pipeline_component import PipeComponent

__all__ = [
    "BaseProcessor",
    "ProcessInfo",
]

ProcessInfo = DataRequest


class BaseProcessor(PipeComponent[PackType], ABC):
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
        self.__is_last_step = False

    # TODO: what if we have config-free processors? It might be cumbersome to
    #  always require a config.
    def initialize(self, resource: Resources, configs: Optional[HParams]):
        """
        The pipeline will call the initialize method at the start of a
        processing. The processor will be initialized with ``configs``,
        and register global resources into ``resource``. The implementation
        should set up the states of the processor.

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

    def set_as_last(self):
        self.__is_last_step = True

    def process(self, input_pack: PackType):
        # Obtain the control of the DataPack.
        input_pack.enter_processing(self.component_name)
        # Do the actual processing.
        self._process(input_pack)

        if not input_pack.is_poison():
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
