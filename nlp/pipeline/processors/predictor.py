from abc import abstractmethod
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.io_utils import merge_batches, slice_batch
from typing import Dict, List, Union, Iterable
from nlp.pipeline.data.base_ontology import BaseOntology
from nlp.pipeline.utils import *


class Predictor(BaseProcessor):
    def __init__(self):
        super().__init__()

        self.current_batch: Dict = {}
        self.instance_num_in_current_batch = 0

        self.data_pack_pool: List[DataPack] = []
        self.current_batch_sources: List[int] = []

    def process(self, input_pack: DataPack, hard_batch=False):
        if hard_batch:
            return self.process_in_hard_batch(input_pack)
        else:
            return self.process_in_soft_batch(input_pack)

    def process_in_soft_batch(self, input_pack: DataPack):
        """
        Process the datapack softly according to batch size. Will process
        as many batches in the input_pack as possible and finish the tail
        instances even if they cannot make up a full batch.

        Args:
            input_pack (DataPack): A datapack to be processed.
        """
        for (data_batch, instance_num) in input_pack.get_data_batch(
                self.batch_size, self.context_type, self.annotation_types):
            pred = self.predict(data_batch)
            self.pack(input_pack, pred)
        self.finish(input_pack)

    def process_in_hard_batch(self, input_pack: DataPack):
        """
        Process the datapack strictly according to batch size. Will process
        as many batches in the input_pack as possible. For the tail instances
        that cannot make up a full batch, will store them in the
        :attr:`current_batch` and process with the next ner_data pack when this
        function is called next time.

        Args:
            input_pack (DataPack): A datapack to be processed.
        """

        self.data_pack_pool.append(input_pack)
        for (data_batch, instance_num) in self.get_data_batch_by_need(
                input_pack, self.context_type, self.annotation_types):

            self.current_batch = merge_batches([self.current_batch, data_batch])
            self.instance_num_in_current_batch += instance_num
            self.current_batch_sources.append(instance_num)

            if self.instance_num_in_current_batch == self.batch_size:
                pred = self.predict(self.current_batch)
                self.pack_all(pred)
                self.finish_up_packs(-1)

                self.current_batch = {}
                self.instance_num_in_current_batch = 0
                self.current_batch_sources = []

        if len(self.current_batch_sources) == 0:
            self.finish_up_packs()

    @abstractmethod
    def predict(self, data_batch: Dict):
        pass

    def pack_all(self, output_dict: Dict):
        start = 0
        for i in range(len(self.data_pack_pool)):
            output_dict_i = slice_batch(output_dict, start,
                                        self.current_batch_sources[i])
            self.pack(self.data_pack_pool[i], output_dict_i)
            start += self.current_batch_sources[i]

    @abstractmethod
    def pack(self, data_pack: DataPack, *inputs) -> None:
        """
        Add corresponding fields to data_pack. Custom function of how
        to add the value back.
        """
        pass

    def finish_up_packs(self, end: int = None):
        """
        Do finishing work for ner_data packs in :attr:`data_pack_pool` from the
        beginning to ``end`` (``end`` is not included).
        """
        if end is None: end = len(self.data_pack_pool)
        for pack in self.data_pack_pool[:end]:
            self.finish(pack)
        self.data_pack_pool = self.data_pack_pool[end:]
        self.current_batch_sources = self.current_batch_sources[end:]

    def finish(self, input_pack: DataPack = None):
        """
        Do finishing work for one data_pack.
        """
        self._record_fields(input_pack)
        input_pack.meta.process_state = get_full_component_name(self)
        # currently, need to build the coverage index after updating the entries
        input_pack.index.build_coverage_index(
            input_pack.annotations,
            input_pack.links,
            input_pack.groups,
            outer_type=BaseOntology.Sentence
        )
        # print(input_pack.links)

    @abstractmethod
    def _record_fields(self, data_pack: DataPack):
        """
        Record the fields and entries that this processor add to ner_data packs.
        """
        pass

    def get_data_batch_by_need(
            self,
            data_pack: DataPack,
            context_type: str,
            annotation_types: Dict[str, Union[Dict, Iterable]] = None,
            link_types: Dict[str, Union[Dict, Iterable]] = None,
            group_types: Dict[str, Union[Dict, Iterable]] = None,
            offset: int = 0) -> Iterable[Dict]:
        """
        Get a ner_data batch from ``data_pack``. If there is enough instances in
        ``data_pack``, the size of the batch is :attr:`batch_size` -
        :attr:`instance_num_in_current_batch`. Otherwise, the size is the
        number of instances left in ``data_pack``.
        """
        batch = {}
        cnt = 0
        for data in data_pack.get_data(context_type, annotation_types,
                                       link_types, group_types, offset):
            for entry, fields in data.items():
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
            cnt += 1
            if cnt == self.batch_size - self.instance_num_in_current_batch:
                yield (batch, cnt)
                cnt = 0
                batch = {}

        if batch:
            yield (batch, cnt)

