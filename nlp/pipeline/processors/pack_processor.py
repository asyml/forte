from abc import abstractmethod
from nlp.pipeline import config
from nlp.pipeline.data import DataPack, PackType
from nlp.pipeline.processors.base_processor import BaseProcessor

__all__ = [
    "BasePackProcessor",
    "PackProcessor",
]


class BasePackProcessor(BaseProcessor[PackType]):
    """
    The base class of processors that process one pack each time.
    """
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


# TODO (Haoran): define MultiPackProcessor
