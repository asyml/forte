from abc import abstractmethod
from typing import (
    Dict, List, Iterator, Iterable, Union, Any, Optional, Tuple, Type)

import texar.torch as tx
from texar.torch import HParams

from forte.data.dataset import Dataset
from forte.data import DataPack, MultiPack, PackType, DataRequest
from forte.data.io_utils import merge_batches, batch_instances
from forte.data.ontology import Entry, Annotation


class Batcher:
    @abstractmethod
    def __init__(self, config: HParams):
        raise NotImplementedError

    @abstractmethod
    def get_batch(self, *args):
        raise NotImplementedError

    @staticmethod
    def default_hparams() -> Dict:
        """
        Defines the basic parameter and values of the batcher.
        :return:
        """
        raise NotImplementedError


class DictData(tx.data.DataBase[Dict, Dict]):
    r"""A dataset that reads processed paired text from dumped NumPy files.

    Args:
        filename (str): The path to the dumped NumPy file.
        hparams: A `dict` or instance of :class:`~texar.HParams` containing
            hyperparameters. See :meth:`default_hparams` for the defaults.
        device: The device of the produces batches. For GPU training, set to
            current CUDA device.
    """

    def __init__(self, dataset: Iterator[Dict], hparams=None):
        data: Iterator[Dict] = dataset
        source = tx.data.IterDataSource(data)
        super().__init__(source, hparams)

    @staticmethod
    def default_hparams():
        return {
            **tx.data.DataBase.default_hparams(),
        }

    def process(self, raw_example: Dict) -> Dict:  # pylint: disable=no-self-use
        return raw_example

    def collate(self,  # pylint: disable=no-self-use
                examples: List[Dict]) -> tx.data.Batch:
        batch = batch_instances(examples)
        return tx.data.Batch(
            len(examples),
            **batch
        )


class ProcessingBatcher(Batcher):
    """
    This defines the basis interface of the Batcher used in BatchProcessors.
    It will create Batches from the packs, and stores the relationship between
    the packs and the Batches, so that we can add the processed result back to
    the packs.
    """

    def __init__(self, batch_size: int, hard_batch: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.hard_batch = hard_batch

        self.current_batch: Dict = {}
        self.instance_num_in_current_batch = 0

        self.data_pack_pool: List[PackType] = []
        self.current_batch_sources: List[int] = []

    def get_batch(  # type: ignore
            self,
            input_pack: Optional[PackType],
            context_type: Type[Annotation],
            requests: DataRequest = None,
            tail_instances: bool = False):

        if input_pack is None:  # No more packs, return the tail instances
            if self.current_batch:
                yield self.current_batch
                self.current_batch = {}
                self.instance_num_in_current_batch = 0
                self.current_batch_sources = []
        else:  # cache the new pack and generate batches
            self.data_pack_pool.append(input_pack)
            for (data_batch, instance_num) in self._get_data_batch_by_need(
                    input_pack, context_type, requests):

                self.current_batch = merge_batches(
                    [self.current_batch, data_batch])

                self.instance_num_in_current_batch += instance_num
                self.current_batch_sources.append(instance_num)

                if (tail_instances or not self.hard_batch or
                        self.instance_num_in_current_batch == self.batch_size):
                    yield self.current_batch

                    self.current_batch = {}
                    self.instance_num_in_current_batch = 0
                    self.current_batch_sources = []

    def _get_data_batch_by_need(
            self,
            data_pack: PackType,
            context_type: Type[Annotation],
            requests: Optional[DataRequest] = None,
            offset: int = 0) -> Iterable[Tuple[Dict, int]]:
        pass


class NumInstancesDataPackBatcher(ProcessingBatcher):
    def _get_data_batch_by_need(
            self,
            data_pack: DataPack,
            context_type: Type[Annotation],
            requests: Optional[Dict[Type[Entry], Union[Dict, List]]] = None,
            offset: int = 0) -> Iterable[Tuple[Dict, int]]:
        """
        Try to get batches of size ``batch_size``. If the tail instances cannot
        make up a full batch, will generate a small batch with the tail
        instances.

        Returns:
            An iterator of tuples ``(batch, cnt)``, ``batch`` is a dict
            containing the required annotations and context, and ``cnt`` is
            the number of instances in the batch.
        """
        instances: List[Dict] = []
        for data in data_pack.get_data(context_type, requests, offset):
            instances.append(data)
            if (len(instances) ==
                    self.batch_size - self.instance_num_in_current_batch):
                batch = batch_instances(instances)
                yield (batch, len(instances))
                instances = []

        if len(instances):
            batch = batch_instances(instances)
            yield (batch, len(instances))


class TexarBatcher(Batcher):
    def __init__(self,
                 data_packs: Iterable[DataPack],
                 context_type: Type[Annotation],
                 batch_size: Optional[int] = None,
                 hparams=None):

        if batch_size is not None:
            hparams["batch_size"] = batch_size
        dataset = Dataset(data_packs)
        data = DictData(dataset.get_data(context_type), hparams=hparams)
        super().__init__()
        self.batch_iter = tx.data.DataIterator(data)

    def get_batch(self) -> Iterator[tx.data.Batch]:  # type: ignore
        for batch in self.batch_iter:
            yield batch

    @staticmethod
    def default_hparams() -> Dict:
        return {
            'context_type': 'forte.ontology.top.Document',
            'batch_size': None
        }


class TxtgenMultiPackProcessingBatcher(NumInstancesDataPackBatcher):
    """
    A Batcher used in ``MultiPackBatchProcessors``.
    The Batcher calles the ProcessingBatcher inherently on each specified
    data pack in the MultiPack.

    It's flexible to query MultiPack so we delegate the task to the subclasses.
    Such as:
    - query all packs with the same ``context`` and ``input_info``.
    - query different packs with different ``context``s and ``input_info``s.
    Since the batcher will save the data_pack_pool on the fly, it's not trivial
        to batching and slicing multiple data packs in the same time
    """

    def __init__(self, input_pack_name: str, batch_size: int,
                 hard_batch: bool = False):
        super().__init__(batch_size)
        super(batch_size, hard_batch)
        self.input_pack_name = input_pack_name

    def _get_data_batch_by_need(
            self,
            data_pack: MultiPack,
            context_type: Type[Annotation],
            requests: Optional[Dict[Type[Entry], Union[Dict, List]]] = None,
            offset: int = 0) -> Iterable[Tuple[Dict, int]]:
        """
        Try to get batches of size ``batch_size``. If the tail instances cannot
        make up a full batch, will generate a small batch with the tail
        instances.

        Returns:
            An iterator of tuples ``(batch, cnt)``, ``batch`` is a dict
            containing the required annotations and context, and ``cnt`` is
            the number of instances in the batch.
        """

        input_pack = data_pack.packs[self.input_pack_name]
        yield from super()._get_data_batch_by_need(
            input_pack, context_type, requests, offset)

    @staticmethod
    def default_hparams() -> Dict:
        pass
