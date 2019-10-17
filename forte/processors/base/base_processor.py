"""
The base class of processors
"""
from abc import abstractmethod, ABC
from typing import Optional

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.data.ontology import base_ontology
from forte.data.selector import DummySelector
from forte.process_manager import ProcessManager
from forte.utils import get_full_module_name
from forte.pipeline_component import PipeComponent

__all__ = [
    "BaseProcessor",
]

process_manager = ProcessManager()


class BaseProcessor(PipeComponent[PackType], ABC):
    """
    The basic processor class. To be inherited by all kinds of processors
    such as trainer, predictor and evaluator.
    """

    def __init__(self):
        self.component_name = get_full_module_name(self)
        self._ontology = base_ontology
        self.selector = DummySelector()

    def initialize(self, resource: Resources, configs: Optional[HParams]):
        """
        The pipeline will call the initialize method at the start of a
        processing. The processor will be initialized with ``configs``,
        and register global resources into ``resource``. The implementation
        should set up the states of the processor.

        Args:
            resource: A global resource register. User can register
             shareable resources here, for example, the vocabulary.
            configs: The configuration passed in to set up this processor.

        Returns:

        """
        pass

    # TODO: remove this.
    def set_ontology(self, ontology):
        """
        Set the ontology of this processor, will be called by the Pipeline.
        """
        self._ontology = ontology  # pylint: disable=attribute-defined-outside-init

    def process(self, input_pack: PackType):
        # Set the component for recording purpose.
        process_manager.set_current_component(self.component_name)
        self._process(input_pack)

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

    def flush(self):
        """
        Indicate that there will be no more packs to be passed in.
        Returns:

        """
        pass

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure
        """
        return {
            'selector': {
                'type': 'forte.data.selector.DummySelector',
                'args': None,
                'kwargs': {}
            }
        }
