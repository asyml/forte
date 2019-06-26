from abc import abstractmethod
from typing import Dict, Any, List
from nlp.pipeline.io.data_pack import DataPack


class BaseProcessor:
    def __init__(self):
        # Initialized model.
        self.context_type = None
        self.annotation_types = None
        self.link_types = None
        self.group_types = None
        self.batch_size = None

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def process(self, *inputs):
        # Do testing or training
        pass

    @abstractmethod
    def pack(self,
             processed_batch: Dict,
             data_packs: List[DataPack],
             start_from: int = 0):
        # Add corresponding fields to data_pack
        for result_key, result_value in processed_batch.items():
            # Custom function of how to add the value back.
            pass
        pass

    @abstractmethod
    def finish(self):
        # Release model.
        pass
