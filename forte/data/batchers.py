from abc import abstractmethod
from typing import (
    Dict, List, Iterator, Iterable, Union, Any, Optional, Tuple, Type)

import texar.torch as tx

from forte.data import Dataset
from forte.data import DataPack, MultiPack
from forte.data import merge_batches, batch_instances
from forte.data import Entry, Annotation


class Batcher:

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def get_batch(self, *args):
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

    def collate(self, examples: List[Dict]) -> tx.data.Batch:  # pylint: disable=no-self-use
        batch: Dict[str, Any] = {}
        for e in examples:
            for entry, fields in e.items():
                if isinstance(fields, dict):
                    if entry not in batch.keys():
                        batch[entry] = {}
                    for k, value in fields.items():
                        if k not in batch[entry].keys():
                            batch[entry][k] = []
                        batch[entry][k].append(value)
                else:  # context level feature
                    if entry not in batch.keys():
                        batch[entry] = []
                    batch[entry].append(fields)
        return tx.data.Batch(
            len(examples),
            **batch
        )


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
        super().__init__(data.batch_size)
        self.batch_iter = tx.data.DataIterator(data)

    def get_batch(self) -> Iterator[tx.data.Batch]:   # type: ignore
        for batch in self.batch_iter:
            yield batch


class ProcessingBatcher(Batcher):
    """
    A Batcher used in BatchProcessors. The Batcher receives new packs
    dynamically and stores the current packs so that the processors can
    pack prediction results into the data packs.
    """

    def __init__(self, batch_size: int, hard_batch: bool = False):
        super().__init__(batch_size)

        self.hard_batch = hard_batch

        self.current_batch: Dict = {}
        self.instance_num_in_current_batch = 0

        self.data_pack_pool: List[DataPack] = []
        self.current_batch_sources: List[int] = []

    def get_batch(  # type: ignore
            self,
            input_pack: Optional[DataPack],
            context_type: Type[Annotation],
            requests: Dict[Type[Entry], Union[Dict, List]] = None,
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
            data_pack: DataPack,
            context_type: Type[Annotation],
            requests: Optional[Dict[Type[Entry], Union[Dict, List]]] = None,
            offset: int = 0) -> Iterable[Tuple[Dict, int]]:
        # TODO: make this abstractmethod. So MultiPack can inherit this class
        # Define another class for SinglePack.
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


class TxtgenMultiPackProcessingBatcher(Batcher):
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

    def __init__(self, input_pack_name, batch_size: int,
                 hard_batch: bool = False,
                 ):
        super().__init__(batch_size)
        self.hard_batch = hard_batch

        self.current_batch: Dict = {}
        self.instance_num_in_current_batch = 0

        self.data_pack_pool: List[MultiPack] = []
        self.current_batch_sources: List[int] = []

        self.input_pack_name = input_pack_name

    def get_batch(  # type: ignore
            self,
            input_pack: Optional[MultiPack],
            context_type: Type[Annotation],
            requests: Dict[Type[Entry], Union[Dict, List]] = None,
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

        instances: List[Dict] = []
        input_pack = data_pack.packs[self.input_pack_name]
        for data in input_pack.get_data(context_type, requests, offset):

            instances.append(data)
            if (len(instances) ==
                    self.batch_size - self.instance_num_in_current_batch):
                batch = batch_instances(instances)
                yield (batch, len(instances))
                instances = []

        if len(instances):
            batch = batch_instances(instances)
            yield (batch, len(instances))
