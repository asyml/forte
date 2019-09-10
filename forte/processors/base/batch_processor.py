from abc import abstractmethod
from typing import Dict, Optional

from forte import config
from forte.data import DataPack, MultiPack, PackType
from forte.data.batchers import ProcessingBatcher, \
    TxtgenMultiPackProcessingBatcher
from forte.data import slice_batch
from forte.data.ontology.top import EntryType
from forte.processors.base.base_processor import BaseProcessor

__all__ = [
    "BaseBatchProcessor",
    "BatchProcessor",
    "MultiPackTxtgenBatchProcessor"
]


class BaseBatchProcessor(BaseProcessor[PackType]):
    """
    The base class of processors that process data in batch.
    """

    def __init__(self):
        super().__init__()

        self.context_type: EntryType = self.define_context()
        self.batch_size = None
        self.batcher = None
        self.use_coverage_index = False

    @abstractmethod
    def define_context(self) -> EntryType:
        """
        User should define the context type for batch processors here
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_batcher(self, hard_batch: bool = True):
        """
        Single pack :class:`BatchProcessor` initialize the batcher to be a
        :class:`ProcessingBatcher`. And MultiPackBatchProcessor might need
        something like "MultiPackProcessingBatcher".
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
    def pack(self, pack: PackType, inputs) -> None:
        """
        Add corresponding fields to pack. Custom function of how
        to add the value back.

        Args:
            pack (PackType): The pack to add entries or fields to.
            inputs: The prediction results returned by :meth:`predict`. You
                need to add entries or fields corresponding to this prediction
                results to the ``data_pack``.
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
        pass


class BatchProcessor(BaseBatchProcessor[DataPack]):
    """
    The batch processors that process DataPacks.
    """

    def initialize_batcher(self, hard_batch: bool = True):
        return ProcessingBatcher(self.batch_size, hard_batch)

    def prepare_coverage_index(self, input_pack: DataPack):
        for entry_type in self.input_info.keys():
            if input_pack.index.coverage_index(self.context_type,
                                               entry_type) is None:
                input_pack.index.build_coverage_index(self.context_type,
                                                      entry_type)


class MultiPackTxtgenBatchProcessor(BaseBatchProcessor[MultiPack]):
    """
    The batch processors that process MultiPack in Txtgen Tasks.
    In this scenario, we don't need to build special bathers since we only need
        to read sentences from one single DataPack
    """

    def __init__(self):
        super().__init__()
        self.input_pack_name = None
        self.output_pack_name = None

    def initialize_batcher(self, hard_batch: bool = True):
        return TxtgenMultiPackProcessingBatcher(self.input_pack_name,
                                                self.batch_size,
                                                hard_batch)

    def prepare_coverage_index(self, input_pack: MultiPack):
        for entry_type in self.input_info.keys():
            if input_pack.packs[self.input_pack_name].index.coverage_index(
                    self.context_type, entry_type) is None:
                input_pack.packs[self.input_pack_name
                ].index.build_coverage_index(self.context_type, entry_type)
