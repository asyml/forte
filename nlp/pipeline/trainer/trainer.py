from abc import abstractmethod
from typing import Dict, Any, Iterator
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.io.data_pack import DataPack


class Trainer(BaseProcessor):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, input_pack: Iterator[DataPack]):
        # Do training
        pass


