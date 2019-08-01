from abc import abstractmethod
from typing import Dict, Optional

from nlp.pipeline import config
from nlp.pipeline.data import slice_batch
from nlp.pipeline.data.base_pack import BasePack
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.data.batchers import ProcessingBatcher

__all__ = [
    "BaseBatchProcessor",
    "BatchProcessor",
    "MultiPackBatchProcessor"
]


class BaseBatchProcessor(BaseProcessor):
    """
    The base class of processors that process data in batch.
    """
    def __init__(self):
        super().__init__()

        self.context_type = None
        self.batch_size = None
        self.batcher = None

    @abstractmethod
    def initialize_batcher(self, hard_batch: bool = True):
        """
        Single pack :class:`BatchProcessor` initialize the batcher to be a
        :class:`ProcessingBatcher`. And MultiPackBatchProcessor might need
        something like "MultiPackProcessingBatcher".
        """
        raise NotImplementedError

    def process(self, input_pack: BasePack, tail_instances: bool = False):
        config.working_component = self.component_name
        if input_pack.meta.cache_state == self.component_name:
            input_pack = None  # type: ignore
        else:
            input_pack.meta.cache_state = self.component_name

        for batch in self.batcher.get_batch(input_pack,
                                            self.context_type,
                                            self.input_info,
                                            tail_instances=tail_instances):
            pred = self.predict(batch)
            self.pack_all(pred)
            self.finish_up_packs(-1)
        if len(self.batcher.current_batch_sources) == 0:
            self.finish_up_packs()
        config.working_component = None

    @abstractmethod
    def predict(self, data_batch: Dict):
        """
        Make predictions for the input data_batch.

        Args:
              data_batch (Dict): A batch of instances in our dict format.

        Returns:
              The prediction results in dict format.
        """
        pass

    def pack_all(self, output_dict: Dict):
        start = 0
        for i in range(len(self.batcher.data_pack_pool)):
            output_dict_i = slice_batch(output_dict, start,
                                        self.batcher.current_batch_sources[i])
            self.pack(self.batcher.data_pack_pool[i], output_dict_i)
            start += self.batcher.current_batch_sources[i]

    @abstractmethod
    def pack(self, data_pack: BasePack, inputs) -> None:
        """
        Add corresponding fields to data_pack. Custom function of how
        to add the value back.

        Args:
            data_pack (BasePack): The data pack to add entries or fields to.
            inputs: The prediction results returned by :meth:`predict`. You
                need to add entries or fields corresponding to this prediction
                results to the ``data_pack``.
        """
        raise NotImplementedError

    def finish_up_packs(self, end: Optional[int] = None):
        """
        Do finishing work for data packs in :attr:`data_pack_pool` from the
        beginning to ``end`` (``end`` is not included).

        Args:
            end (int): Will do finishing work for data packs in
                :attr:`data_pack_pool` from the beginning to ``end``
                (``end`` is not included). If `None`, will finish up all the
                packs in :attr:`data_pack_pool`.
        """
        if end is None:
            end = len(self.batcher.data_pack_pool)
        for pack in self.batcher.data_pack_pool[:end]:
            self.finish(pack)
        self.batcher.data_pack_pool = self.batcher.data_pack_pool[end:]
        self.batcher.current_batch_sources = \
            self.batcher.current_batch_sources[end:]


class BatchProcessor(BaseBatchProcessor):
    """
    The batch processors that process DataPacks.
    """
    def initialize_batcher(self, hard_batch: bool = True):
        self.batcher = ProcessingBatcher(self.batch_size, hard_batch)


class MultiPackBatchProcessor(BaseBatchProcessor):
    """
        The batch processors that process MultiPacks.
    """
    def initialize_batcher(self, hard_batch: bool = True):
        pass
