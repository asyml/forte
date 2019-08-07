from abc import abstractmethod
from typing import Dict, Optional

from nlp.pipeline import config
from nlp.pipeline.data import PackType, DataPack
from nlp.pipeline.data.io_utils import slice_batch
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.data.batchers import ProcessingBatcher

__all__ = [
    "BaseBatchProcessor",
    "BatchProcessor",
]


class BaseBatchProcessor(BaseProcessor[PackType]):
    """
    The base class of processors that process data in batch.
    """
    def __init__(self):
        super().__init__()

        self.context_type = None
        self.batch_size = None
        self.batcher = None
        self.use_coverage_index = False

    @abstractmethod
    def initialize_batcher(self, hard_batch: bool = True):
        """
        Single pack :class:`BatchProcessor` initialize the batcher to be a
        :class:`~nlp.pipeline.data.batchers.ProcessingBatcher`.
        And :class:`MultiPackBatchProcessor` initialize the batcher to be a
        :class:`~nlp.pipeline.data.batchers.MultiPackProcessingBatcher`.
        """
        raise NotImplementedError

    def process(self, input_pack: PackType, tail_instances: bool = False):
        config.working_component = self.component_name
        if input_pack.meta.cache_state == self.component_name:
            input_pack = None  # type: ignore
        else:
            input_pack.meta.cache_state = self.component_name

        if self.use_coverage_index:
            self.prepare_coverage_index(input_pack)
        for batch in self.batcher.get_batch(input_pack,
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
        The function that task processors should implement.

        Make predictions for the input ``data_batch``.

        Args:
              data_batch (dict): A batch of instances in our dict format.

        Returns:
              The prediction results in dict format.
        """
        pass

    def pack_all(self, output_dict: Dict):
        """
        Pack the prediction results ``output_dict`` back to the
        corresponding packs.
        """
        start = 0
        for i in range(len(self.batcher.data_pack_pool)):
            output_dict_i = slice_batch(output_dict, start,
                                        self.batcher.current_batch_sources[i])
            self.pack(self.batcher.data_pack_pool[i], output_dict_i)
            start += self.batcher.current_batch_sources[i]

    @abstractmethod
    def pack(self, pack: PackType, output_dict: Dict) -> None:
        """
        The function that task processors should implement.

        Add corresponding fields to ``pack``. Custom function of how
        to add the value back.

        Args:
            pack (PackType): The pack to add entries or fields to.
            output_dict: The prediction results returned by :meth:`predict`. You
                need to add entries or fields corresponding to this prediction
                results to ``pack``.
        """
        raise NotImplementedError

    def finish_up_packs(self, end: Optional[int] = None):
        """
        Do finishing work for packs in :attr:`data_pack_pool` from the
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

    @abstractmethod
    def prepare_coverage_index(self, input_pack: PackType):
        """
        Build the coverage index for ``input_pack`` according to
        :attr:`input_info`.
        """
        pass


class BatchProcessor(BaseBatchProcessor[DataPack]):
    """
    The batch processors that process :class:`DataPack`.
    """
    def initialize_batcher(self, hard_batch: bool = True):
        return ProcessingBatcher(self.batch_size, self.context_type,
                                 self.input_info, hard_batch)

    def prepare_coverage_index(self, input_pack: DataPack):
        for entry_type in self.input_info.keys():
            if input_pack.index.coverage_index(self.context_type,
                                               entry_type) is None:
                input_pack.index.build_coverage_index(self.context_type,
                                                      entry_type)


# TODO (Haoran): define MultiPackBatchProcessor
