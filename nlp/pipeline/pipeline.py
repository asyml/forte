from typing import List, Dict
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.io.data_pack import DataPack
from nlp.pipeline.io.readers.ontonotes_reader import OntonotesReader
from nlp.pipeline.io.readers.base_reader import BaseReader


class Pipeline:
    def __init__(self, **kwargs):
        self.reader = None
        self.dataset_iterator = None  # or put this into Reader class

        self.processors: List[BaseProcessor] = []

        self.current_packs: List[DataPack] = []
        self._tail_instances = 0

        self._config(**kwargs)

    def _config(self, **kwargs):
        if "dataset" in kwargs.keys():
            dataset_dir = kwargs["dataset"]["dataset_dir"]
            dataset_format = kwargs["dataset"]["dataset_format"]

            if dataset_format.lower() == "ontonotes":
                self.reader = OntonotesReader()
            else:
                self.reader = BaseReader()

            self.dataset_iterator = self.reader.dataset_iterator(dataset_dir)

    def _load_datapacks_for_next_batch(self,
                                       batch_size: int,
                                       count_sentence: bool = True):

        if self._tail_instances == 0:
            self.current_packs.clear()
            start_from = 0
        else:
            self.current_packs = self.current_packs[-1:]
            start_from = self.current_packs[0].internal_metas[
                             "Sentence"
                         ].id_counter - self._tail_instances  # sentence component

        instance_cnt = self._tail_instances

        while instance_cnt < batch_size:
            try:
                data_pack = next(self.dataset_iterator)
            except StopIteration:
                break
            self.current_packs.append(data_pack)
            if count_sentence:
                # TODO: need to ensure all processor request
                #  sentence from the same component?
                instance_num = data_pack.internal_metas["Sentence"].id_counter
            else:
                instance_num = 1
            instance_cnt += instance_num

        self._tail_instances = instance_cnt - batch_size

        return start_from

    def _get_batch_as_numpy(self,
                            batch_size: int,
                            start_from: int,
                            context_type: str,
                            annotation_types: Dict,
                            link_types: Dict,
                            group_types: Dict
                            ) -> Dict:
        batch = dict()
        instance_cnt = 0
        for pack in self.current_packs:
            instances = list(pack.get_data(context_type,
                                           annotation_types,
                                           link_types,
                                           group_types))
            for data in instances[start_from:]:
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
                instance_cnt += 1
                if instance_cnt == batch_size:
                    return batch
            start_from = 0

        return batch

    def process_next(self, batch_size: int = 1):

        start_from = self._load_datapacks_for_next_batch(batch_size)

        for processor in self.processors:
            batch = self._get_batch_as_numpy(batch_size,
                                             start_from,
                                             processor.context_type,
                                             processor.annotation_types,
                                             processor.link_types,
                                             processor.group_types)
            results = processor.process(batch)
            processor.pack(results, self.current_packs, start_from)

        #write out




