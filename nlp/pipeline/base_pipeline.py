from abc import abstractmethod
from typing import List, Dict

from nlp.pipeline.data.readers import BaseReader
from nlp.pipeline.processors import BaseProcessor

__all__ = [
    "BasePipeline"
]


class BasePipeline:
    """
    The pipeline consists of a list of predictors.
    """

    def __init__(self, **kwargs):
        self._reader: BaseReader = BaseReader()
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
        if "ontology" in kwargs.keys():
            self._ontology = kwargs["ontology"]
            self._reader.set_ontology(self._ontology)
            for processor in self.processors:
                processor.set_ontology(self._ontology)

    def set_reader(self, reader: BaseReader):
        if self._ontology is not None:
            reader.set_ontology(self._ontology)
        self._reader = reader

    @property
    def processors(self):
        return self._processors

    @abstractmethod
    def add_processor(self, processor: BaseProcessor):
        raise NotImplementedError

    @abstractmethod
    def process(self, data):
        raise NotImplementedError

    @abstractmethod
    def process_dataset(self, dataset: str):
        """
        Process the documents in the dataset and return an iterator of DataPack.

        Args:
            dataset (str): the directory of the dataset to be processed.
        """
        raise NotImplementedError
