from abc import abstractmethod
from typing import List, Dict, Iterator, Generic

from nlp.pipeline.data.base_pack import PackType
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.data.readers import BaseReader
from nlp.pipeline.processors import BaseProcessor

__all__ = [
    "BasePipeline"
]


class BasePipeline(Generic[PackType]):
    """
    The pipeline consists of a list of predictors.
    TODO(Wei): check fields when concatenating processors
    """

    def __init__(self, **kwargs):
        self._reader: BaseReader = None
        self._processors: List[BaseProcessor] = []
        self._processors_index: Dict = {'': -1}

        self._ontology = base_ontology
        self.topology = None
        self.current_packs = []

        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        """
        Initialize the pipeline with configs
        """
        if "ontology" in kwargs.keys():
            self._ontology = kwargs["ontology"]
            if self._reader is not None:
                self._reader.set_ontology(self._ontology)
            for processor in self.processors:
                processor.set_ontology(self._ontology)

    def set_reader(self, reader: BaseReader):
        reader.set_ontology(self._ontology)
        self._reader = reader

    @property
    def processors(self):
        return self._processors

    @abstractmethod
    def add_processor(self, processor: BaseProcessor):
        raise NotImplementedError

    @abstractmethod
    def process(self, data: str) -> PackType:
        """
        Process a string text or a single file.

        Args:
            data (str): the path to a file a string text. If :attr:`_reader` is
                :class:`StringReader`, `data` should be a text in the form of
                a string variable. If :attr:`_reader` is a file reader, `data`
                should be the path to a file.
        """
        raise NotImplementedError

    @abstractmethod
    def process_dataset(self, dataset: str) -> Iterator[PackType]:
        """
        Process the documents in the dataset and return an iterator of DataPack.

        Args:
            dataset (str): the directory of the dataset to be processed.
        """
        raise NotImplementedError
