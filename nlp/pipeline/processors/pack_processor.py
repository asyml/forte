from abc import abstractmethod

from nlp.pipeline import config
from nlp.pipeline.data import BasePack
from nlp.pipeline.processors.base_processor import BaseProcessor

__all__ = [
    "BasePackProcessor",
    "PackProcessor",
    "MultiPackProcessor"
]


class BasePackProcessor(BaseProcessor):
    """
    The base class of processors that process one pack each time.
    """
    def process(self, input_pack: BasePack):
        """
        Process one datapack at a time.

        Args:
            input_pack (BasePack): A datapack to be processed.
        """
        config.working_component = self.component_name
        self._process(input_pack)
        self.finish(input_pack)
        config.working_component = None

    @abstractmethod
    def _process(self, input_pack: BasePack):
        pass


class PackProcessor(BasePackProcessor):
    """
    The base class of processors that process one pack each time.
    """
    pass


class MultiPackProcessor(BasePackProcessor):
    """
    The base class of processors that process one pack each time.
    """
    pass
