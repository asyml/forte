from abc import abstractmethod, ABC
from typing import Dict, Optional, Type

from texar.torch import HParams

from forte.common import Resources
from forte.data.base_pack import PackType
from forte.data import DataPack, MultiPack
from forte.data import slice_batch
from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcher
from forte.data.ontology.top import Annotation
from forte.processors.base.base_processor import BaseProcessor

__all__ = [
    "BaseBatchProcessor",
    "BatchProcessor",
    "MultiPackBatchProcessor",
    "FixedSizeBatchProcessor",
    "FixedSizeMultiPackBatchProcessor"
]


class BaseBatchProcessor(BaseProcessor[PackType], ABC):
    """
    The base class of processors that process data in batch. This processor
    enables easy data batching via analyze the context and data objects. The
    context defines the scope of analysis of a particular task. For example, in
    dependency parsing, the context is normally a sentence, in entity
    coreference, the context is normally a document. The processor will create
    data batches relative to the context.
    """

    def __init__(self):
        super().__init__()
        self.context_type: Type[Annotation] = self.define_context()
        self.batcher: ProcessingBatcher = self.define_batcher()
        self.use_coverage_index = False

    def initialize(self, resource: Resources, configs: Optional[HParams]):
        super().initialize(resource, configs)
        # Initialize the batcher.
        self.batcher.initialize(configs)

    @abstractmethod
    def define_context(self) -> Type[Annotation]:
        """
        User should define the context type for batch processors here. The
        context must be of type :class:`Annotation`, the processor will create
        data batches with in the span of each annotations. For example, if the
        context type is ``Sentence``, and the task is POS tagging, then each
        batch will contain the POS tags for all words in the sentence.

        The "context" parameter here has the same meaning as the
        :meth:``get_data()`` function in class :class:``DataPack``.
        """
        raise NotImplementedError

    @abstractmethod
    def define_batcher(self) -> ProcessingBatcher:
        """
        Define a specific batcher for this processor.
        Single pack :class:`BatchProcessor` initialize the batcher to be a
        :class:`~forte.data.batchers.ProcessingBatcher`.
        And :class:`MultiPackBatchProcessor` initialize the batcher to be a
        :class:`~forte.data.batchers.MultiPackProcessingBatcher`.
        """
        raise NotImplementedError

    def _process(self, input_pack: PackType):
        """
        In batch processors, all data are processed in batches. So this function
        is implemented to convert the input datapacks into batches according to
        the Batcher. Users do not need to implement this function but should
        instead implement ``predict``, which computes results from batches, and
        ``pack``, which convert the batch results back to datapacks.

        Args:
            input_pack: The next input pack to be fed in.

        Returns:

        """
        if input_pack.meta.cache_state == self.component_name:
            input_pack = None  # type: ignore
        else:
            input_pack.meta.cache_state = self.component_name

        if self.use_coverage_index:
            self.prepare_coverage_index(input_pack)
        for batch in self.batcher.get_batch(
                input_pack, self.context_type, self.input_info):
            pred = self.predict(batch)
            self.pack_all(pred)
            self.finish_up_packs(-1)
        if len(self.batcher.current_batch_sources) == 0:
            self.finish_up_packs()

    @abstractmethod
    def predict(self, data_batch: Dict):
        """
        The function that task processors should implement.

        Make predictions for the input ``data_batch``.

        Args:
              data_batch (dict): A batch of instances in our dict format.

        Returns:
              The prediction results in dict datasets.
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
    def pack(self, pack: PackType, inputs) -> None:
        """
        The function that task processors should implement.

        Add corresponding fields to ``pack``. Custom function of how
        to add the value back.

        Args:
            pack (PackType): The pack to add entries or fields to.
            inputs: The prediction results returned by :meth:`predict`. You
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


class BatchProcessor(BaseBatchProcessor[DataPack], ABC):
    """
    The batch processors that process :class:`DataPack`.
    """

    def prepare_coverage_index(self, input_pack: DataPack):
        for entry_type in self.input_info.keys():
            if input_pack.index.coverage_index(self.context_type,
                                               entry_type) is None:
                input_pack.index.build_coverage_index(
                    input_pack,
                    self.context_type,
                    entry_type
                )


class FixedSizeBatchProcessor(BatchProcessor, ABC):
    # pylint: disable=no-self-use
    def define_batcher(self) -> ProcessingBatcher:
        return FixedSizeDataPackBatcher()


class MultiPackBatchProcessor(BaseBatchProcessor[MultiPack], ABC):
    """
    This just defines the generic type to MultiPack.
    The implemented batch processors will process MultiPacks.
    """

    def __init__(self):
        super().__init__()
        self.input_pack_name = None

    def prepare_coverage_index(self, input_pack: MultiPack):
        for entry_type in self.input_info.keys():
            if input_pack.packs[self.input_pack_name].index.coverage_index(
                    self.context_type, entry_type) is None:
                input_pack.packs[
                    self.input_pack_name].index.build_coverage_index(
                    self.context_type, entry_type)


class FixedSizeMultiPackBatchProcessor(MultiPackBatchProcessor, ABC):
    # pylint: disable=no-self-use
    def define_batcher(self) -> ProcessingBatcher:
        return FixedSizeDataPackBatcher()
