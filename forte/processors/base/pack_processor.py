from abc import abstractmethod, ABC
from forte import config
from forte.data import DataPack, PackType, MultiPack
from forte.processors.base.base_processor import BaseProcessor

__all__ = [
    "BasePackProcessor",
    "PackProcessor",
]


class BasePackProcessor(BaseProcessor[PackType], ABC):
    """
    The base class of processors that process one pack sequentially. If you are
    looking for batching (that might happen across packs, refer to
    BaseBatchProcessor.
    """

    def process(self, input_pack: PackType):
        """
        Process one datapack at a time.

        Args:
            input_pack (PackType): A datapack to be processed.
        """
        self._process(input_pack)


class PackProcessor(BasePackProcessor[DataPack], ABC):
    """
    The base class of processors that process one pack each time.
    """
    pass


class MultiPackProcessor(BasePackProcessor[MultiPack], ABC):
    """
    The base class of processors that process MultiPack each time
    """
    pass
