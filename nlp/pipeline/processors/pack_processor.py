from abc import abstractmethod

from nlp.pipeline.processors import BaseProcessor
from nlp.pipeline.data.data_pack import DataPack

__all__ = [
    "PackProcessor",
]


class PackProcessor(BaseProcessor):
    """
    The base class of processors that process one pack each time.
    """

    def __init__(self):
        super().__init__()
        self._overwrite = True

    def set_mode(self, overwrite: bool):
        self._overwrite = overwrite

    def process(self, input_pack: DataPack):
        """
        Process one datapack at a time.

        Args:
            input_pack (DataPack): A datapack to be processed.
        """
        self._process(input_pack)
        self.finish(input_pack)

    @abstractmethod
    def _process(self, input_pack: DataPack):
        pass
