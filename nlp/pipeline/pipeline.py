from typing import List, Dict, Tuple, Iterator
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.io.data_pack import DataPack
from nlp.pipeline.io.readers.ontonotes_reader import OntonotesReader
from nlp.pipeline.io.readers.base_reader import BaseReader


class Pipeline:
    def __init__(self, **kwargs):
        self.reader = None
        self.dataset_iterator = None  # or put this into Reader class

        self.processors: List[BaseProcessor] = []
        self._processors_beginning: List[Tuple[int, int]] = []
        self.current_packs: List[DataPack] = []

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

    def _load_next_datapacks(self,
                             instance_need: int,
                             instance_level: str = "sentence"):
        """
        Load new data packs into `current_datapacks` according to the request.

        Args:
            instance_need: the number of instances needed.
            instance_level: the level (granularity) of instances needed.
                Will count instances in this level.
        """

        instance_cnt = 0

        while instance_cnt < instance_need:
            try:
                data_pack = next(self.dataset_iterator)
            except StopIteration:
                break  # need to deal with stop iteration exception
            self.current_packs.append(data_pack)
            if instance_level == "sentence":
                instance_num = data_pack.internal_metas["Sentence"].id_counter
            elif instance_level == "document":
                instance_num = 1
            else:  # need to add other instance levels
                raise ValueError(f"Invalid instance level. Should be 'document'"
                                 f" or 'sentence'.")
            instance_cnt += instance_num

    def _get_batch_as_numpy(self, processor, processor_index) -> Dict:
        batch = dict()
        instance_cnt = 0
        pack_offset, instance_offset = \
            self._processors_beginning[processor_index]

        if not self.current_packs:
            self._load_next_datapacks(processor.batch_size,
                                      processor.context_type)

        for pack_index, pack in enumerate(self.current_packs[pack_offset:],
                                          pack_offset):

            instances = pack.get_data(processor.context_type,
                                      processor.annotation_types,
                                      processor.link_types,
                                      processor.group_types,
                                      instance_offset)
            for data in instances:
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
                instance_offset += 1
                if instance_cnt == processor.batch_size:
                    self._processors_beginning[processor_index] = (
                        pack_offset, instance_offset
                    )
                    return batch

            pack_offset += 1
            instance_offset = 0

            if pack_offset == len(self.current_packs):
                self._load_next_datapacks(processor.batch_size - instance_cnt,
                                          processor.context_type)

        self._processors_beginning[processor_index] = (
            pack_offset, instance_offset
        )

        return batch

    def process_next(self) -> Iterator[DataPack]:
        for pack in self.dataset_iterator:
            for processor_index, processor in enumerate(self.processors):
                processor.process(pack)
            yield pack
            # write out


