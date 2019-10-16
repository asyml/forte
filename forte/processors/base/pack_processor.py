from abc import abstractmethod, ABC

from forte import config
from forte.data.base_pack import PackType
from forte.data import DataPack, MultiPack
from forte.processors.base.base_processor import BaseProcessor

__all__ = [
    "BasePackProcessor",
    "PackProcessor",
    "MultiPackProcessor"
]


class BasePackProcessor(BaseProcessor[PackType], ABC):
    """
    The base class of processors that process one pack sequentially. If you are
    looking for batching (that might happen across packs, refer to
    BaseBatchProcessor.
    """

    def process(self, input_pack: PackType):
        """
        Process one `input_pack` at a time.

        Args:
            input_pack (PackType): A pack to be processed.
        """
        if input_pack.is_poison():
            return

        self._process(input_pack)
        config.working_component = None

    @abstractmethod
    def _process(self, input_pack: PackType):
        """
        The function that task processors should implement.

        In this function, the processor process ``input_pack`` and add new
        entries and fields to the ``input_pack``.
        """
        pass


class PackProcessor(BasePackProcessor[DataPack], ABC):
    """
    The base class of processors that process one :class:`DataPack` each time.
    """
    pass


class MultiPackProcessor(BasePackProcessor[MultiPack], ABC):
    """
    The base class of processors that process MultiPack each time
    """
    pass
