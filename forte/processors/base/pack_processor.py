from abc import ABC

from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
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
