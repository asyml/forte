from abc import abstractmethod
from nlp.pipeline.io.data_pack import DataPack
from typing import List, Dict

class BaseProcessor:
    def __init__(self):
        # Initialized model.
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def process(self, data_batch: Dict):
        # Do testing, return new list of results
        pass

    @abstractmethod
    def pack(self, processed_batch: Dict, data_pack: DataPack):
        # Add corresponding fields to data_pack
        for result_key, result_value in processed_batch.items():
            # Custom function of how to add the value back.
            pass
        pass

    @abstractmethod
    def finish(self):
        # Release model.
        pass




