from abc import abstractmethod
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.data.data_pack import DataPack
from typing import Dict

class Predictor(BaseProcessor):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, input_pack: DataPack) -> None:
        # Do testing, return a dict of results
        pass

    @abstractmethod
    def pack(self, data_pack: DataPack, *inputs):
        """Add corresponding fields to data_pack. Custom function of how
        to add the value back.
        """
        pass


