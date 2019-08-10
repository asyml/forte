from abc import abstractmethod
from forte import config
from forte.data import DataPack, PackType, MultiPack
from forte.processors.base.base_processor import BaseProcessor

__all__ = [
    "BasePackProcessor",
    "PackProcessor",
]


class BasePackProcessor(BaseProcessor[PackType]):
    """
    The base class of processors that process one pack each time.
    """

    def set_ontology(self, ontology):
        self._ontology = ontology  # pylint: disable=attribute-defined-outside-init
        self.define_input_info()
        self.define_output_info()

    def process(self, input_pack: PackType):
        """
        Process one datapack at a time.

        Args:
            input_pack (PackType): A datapack to be processed.
        """
        config.working_component = self.component_name
        self._process(input_pack)
        self.finish(input_pack)
        config.working_component = None

    @abstractmethod
    def _process(self, input_pack: PackType):
        pass


class PackProcessor(BasePackProcessor[DataPack]):
    """
    The base class of processors that process one pack each time.
    """
    pass


class MultiPackProcessor(BasePackProcessor[MultiPack]):
    """
    The base class of processors that process MultiPack each time
    """
    pass
