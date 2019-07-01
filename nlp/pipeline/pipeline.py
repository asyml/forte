from typing import List, Iterator
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.readers.ontonotes_reader import OntonotesReader
from nlp.pipeline.data.readers.conll03_reader import CoNLL03Reader
from nlp.pipeline.data.readers.base_reader import BaseReader
from nlp.pipeline.processors.predictor import Predictor
from nlp.pipeline.utils import *


class Pipeline:
    """
    The pipeline consists of a list of predictors.
    """
    def __init__(self, **kwargs):
        self.reader = None
        self.dataset_dir = None
        self.processors: List[Predictor] = []

        self.topology = None
        self.current_packs = []

        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        """
        Initialize the pipeline with configs
        """
        if "dataset" in kwargs.keys():
            self.dataset_dir = kwargs["dataset"]["dataset_dir"]
            dataset_format = kwargs["dataset"]["dataset_format"]

            if dataset_format.lower() == "ontonotes":
                self.reader = OntonotesReader()
            elif dataset_format.lower() == "conll03":
                self.reader = CoNLL03Reader()
            else:
                self.reader = BaseReader()

    def run(self, hard_batch: bool = True) -> Iterator[DataPack]:
        """
        Process the documents in the dataset and return an iterator of DataPack.

        Args:
            hard_batch (bool): Determines whether to process the dataset
                strictly according to batch_size. (This will only influence
                the efficiency of this method, but will not change the
                result. For small datapacks, using hard batch should be more
                time-saving; for large datapacks (avg instance num in each
                datapack >> batch size), using soft batch is more
                space-saving.)
        """
        if hard_batch:
            yield from self._process_next_in_hard_batch()
        else:
            yield from self._process_next_in_soft_batch()

    def _process_next_in_soft_batch(self) -> Iterator[DataPack]:
        for pack in self.reader.dataset_iterator(self.dataset_dir):
            for processor_index, processor in enumerate(self.processors):
                processor.process(pack, hard_batch=False)
            yield pack

    def _process_next_in_hard_batch(self) -> Iterator[DataPack]:
        for pack in self.reader.dataset_iterator(self.dataset_dir):
            self.current_packs.append(pack)
            for i, processor in enumerate(self.processors):
                for c_pack in self.current_packs:
                    if (i == 0 or c_pack.meta.process_state ==
                            get_full_component_name(self.processors[i - 1])):
                        processor.process(c_pack, hard_batch=True)
                for c_pack in self.current_packs:
                    if c_pack.meta.process_state == get_full_component_name(
                                self.processors[-1]):
                        yield c_pack
                        self.current_packs.remove(c_pack)
        # process tail instances in the whole dataset
        for c_pack in self.current_packs:
            for processor_index, processor in enumerate(self.processors):
                processor.process(c_pack, hard_batch=False)
            yield c_pack
            self.current_packs.remove(c_pack)

