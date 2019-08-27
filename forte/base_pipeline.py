from abc import abstractmethod
from typing import List, Dict, Iterator, Generic, Optional
import logging

import yaml
from texar.torch import HParams

from forte.data.base_pack import PackType
from forte.data.ontology import base_ontology
from forte.data.readers import BaseReader
from forte.processors.base import BaseProcessor, BaseBatchProcessor
from forte.common.resources import Resources

logger = logging.getLogger(__name__)

__all__ = [
    "BasePipeline"
]


class BasePipeline(Generic[PackType]):
    """
    The pipeline consists of a list of predictors.
    TODO(Wei): check fields when concatenating processors
    """

    def __init__(self):
        self._reader: BaseReader = None
        self._processors: List[BaseProcessor] = []
        self._processors_index: Dict = {'': -1}
        self._configs: List[Optional[HParams]] = []

        self._ontology = base_ontology
        self.topology = None
        self.current_packs = []
        self.resource = Resources()

    def init_from_config_path(self, config_path):
        """
        Read the configs from the given path ``config_path``
        and initialize the pipeline including processors
        """
        # TODO: Typically, we should also set the reader here
        # This will be done after StringReader is merged
        # We need to modify the read -> read_file_as_pack then.
        configs = yaml.safe_load(open(config_path))

        self.init_from_config(configs)

    @abstractmethod
    def init_from_config(self, configs: Dict):
        """
        Inittialized the pipeline (ontology and processors) from given configs
        """
        raise NotImplementedError

    def set_ontology(self, ontology):
        self._ontology = ontology
        for processor in self.processors:
            processor.set_ontology(self._ontology)

    def initialize_processors(self):
        for processor, config in zip(self.processors, self.processor_configs):
            processor.initialize(config, self.resource)
            processor.set_ontology(self._ontology)
            processor.set_input_info()
            processor.set_output_info()

    def set_reader(self, reader: BaseReader):
        reader.set_ontology(self._ontology)
        self._reader = reader

    @property
    def processors(self):
        return self._processors

    @property
    def processor_configs(self):
        return self._configs

    def add_processor(self,
                      processor: BaseProcessor,
                      config: Optional[HParams] = None):
        if self._ontology:
            processor.set_ontology(self._ontology)
        self._processors_index[processor.component_name] = len(self.processors)
        self.processors.append(processor)
        self.processor_configs.append(config)

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

    def process_dataset(self, dataset: str) -> Iterator[PackType]:
        """
        Process the documents in the dataset and return an iterator of DataPack.

        Args:
            dataset (str): the directory of the dataset to be processed.

        """

        data_iter = self._reader.iter(dataset)

        if len(self.processors) == 0:
            yield from data_iter

        else:
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
                    # must iterate through a copy of the original list
                    # because of the removing operation
                    if (c_pack.meta.process_state ==
                            self.processors[-1].component_name):
                        yield c_pack
                        self.current_packs.remove(c_pack)

            # process tail instances in the whole dataset
            for c_pack in list(self.current_packs):
                start = self._processors_index[c_pack.meta.process_state] + 1
                for processor in self.processors[start:]:
                    if isinstance(processor, BaseBatchProcessor):
                        processor.process(c_pack, tail_instances=True)
                    else:
                        processor.process(c_pack)
                yield c_pack
                self.current_packs.remove(c_pack)
