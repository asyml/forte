from typing import Iterator, List, Optional, Dict

from nlp.pipeline.data import DataPack
from nlp.pipeline.data.readers import (
    CoNLL03Reader, OntonotesReader, PlainTextReader)
from nlp.pipeline.processors import BaseProcessor, BatchProcessor
from nlp.pipeline.utils import get_class


class Pipeline:
    """
    The pipeline consists of a list of predictors.
    """

    def __init__(self, **kwargs):
        self.reader = None
        self.dataset_dir = None
        self._processors: List[BaseProcessor] = []
        self._processors_index: Dict = {'': -1}

        self._ontology = None
        self.topology = None
        self.current_packs = []

        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        """
        Initialize the pipeline with configs
        """
        if "dataset" in kwargs.keys():
            self.initialize_dataset(kwargs["dataset"])
        if "ontology" in kwargs.keys():
            module_path = ["__main__",
                           "nlp.pipeline.data.ontology"]
            self._ontology = get_class(kwargs["ontology"], module_path)
            for processor in self.processors:
                processor.ontology = self._ontology

    def initialize_dataset(self, dataset):
        self.dataset_dir = dataset["dataset_dir"]
        dataset_format = dataset["dataset_format"]

        if dataset_format.lower() == "ontonotes":
            self.reader = OntonotesReader()
        elif dataset_format.lower() == "conll03":
            self.reader = CoNLL03Reader()
        else:
            self.reader = PlainTextReader()

    @property
    def processors(self):
        return self._processors

    def add_processor(self, processor: BaseProcessor):
        if self._ontology:
            processor.ontology = self._ontology
        self._processors_index[processor.component_name] = len(self.processors)
        self.processors.append(processor)

    def process(self, text: str):
        datapack = DataPack()
        datapack.text = text
        for processor in self.processors:
            if isinstance(processor, BatchProcessor):
                processor.process(datapack, tail_instances=True)
            else:
                processor.process(datapack)
        return datapack

    def process_dataset(self, dataset: Optional[dict] = None) -> Iterator[DataPack]:
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

        if isinstance(dataset, dict):
            self.initialize_dataset(dataset)
            data_iter = self.reader.dataset_iterator(self.dataset_dir)
        elif isinstance(dataset, str) and self.reader is not None:
            self.dataset_dir = dataset
            data_iter = self.reader.dataset_iterator(dataset)
        elif dataset is None and self.reader and self.dataset_dir:
            data_iter = self.reader.dataset_iterator(self.dataset_dir)
        else:
            raise ValueError("Please specify the path to the dataset")

        for pack in data_iter:
            self.current_packs.append(pack)
            for i, processor in enumerate(self.processors):
                for c_pack in self.current_packs:
                    in_cache = (c_pack.meta.cache_state ==
                                processor.component_name)
                    can_process = (i == 0 or c_pack.meta.process_state ==
                                   self.processors[i - 1].component_name)
                    if can_process and not in_cache:
                        processor.process(c_pack)
            for c_pack in list(self.current_packs):
                # must iterate through a copy of the originial list
                # because of the removing operation
                if (c_pack.meta.process_state ==
                        self.processors[-1].component_name):
                    yield c_pack
                    self.current_packs.remove(c_pack)

        # process tail instances in the whole dataset
        for c_pack in list(self.current_packs):
            start = self._processors_index[c_pack.meta.process_state] + 1
            for processor in self.processors[start:]:
                if isinstance(processor, BatchProcessor):
                    processor.process(c_pack, tail_instances=True)
                else:
                    processor.process(c_pack)
            yield c_pack
            self.current_packs.remove(c_pack)
