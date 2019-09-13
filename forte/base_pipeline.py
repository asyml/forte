from abc import abstractmethod
from typing import List, Dict, Iterator, Generic, Optional, Union
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
        self.__working_component: str = None

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
        Initialized the pipeline (ontology and processors) from given configs
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

        self.__working_component = None

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

    def process(self, *args, **kwargs) -> PackType:
        """
        Alias for process_one.
        Args:
            *args:
            **kwargs:

        Returns:

        """
        return self.process_one(*args, **kwargs)

    def run(self, *args, **kwargs):
        """
        Run the whole pipeline and ignore all returned DataPack. This is used
        when the users are relying on the side effect of the processors (e.g.
        a process that will write Packs to disk).
        Args:
            *args:
            **kwargs:

        Returns:

        """
        for _ in self.process_dataset(*args, **kwargs):
            # Process the whole dataset ignoring the return values.
            # This essentially expect the processors have side effects.
            pass

    def process_one(self, *args, **kwargs) -> PackType:
        """
        Process one single data pack. This is done by only reading and
        processing the first pack in the reader.

        Args:
            kwargs: the information needed to load the data. For example, if
                :attr:`_reader` is :class:`StringReader`, this should contain a
                single piece of text in the form of a string variable. If
                :attr:`_reader` is a file reader, this can point to the file
                path.
        """
        first_pack = []
        for p in self._reader.iter(*args, **kwargs):
            first_pack.append(p)
            break

        if len(first_pack) == 1:
            results = [p for p in self.process_packs(first_pack)]
            return results[0]
        else:
            raise ValueError("Input data source contains no packs.")

    def process_dataset(self, *args, **kwargs) -> \
            Union[Iterator[PackType], List[PackType]]:
        """
        Process the documents in the data source(s) and return an
        iterator or list of DataPacks.

        Args:
            **kwargs, which can be one or more data sources.
        """
        data_iter = self._reader.iter(*args, **kwargs)
        return self.process_packs(data_iter)

    def process_packs(
            self, data_iter: Union[Iterator[PackType], List[PackType]]
    ) -> Iterator[PackType]:
        """
        Process an iterator of data packs and return the  processed ones.
        Args:
            data_iter: An iterator of the data packs.

        Returns: A list data packs.

        """
        if len(self.processors) == 0:
            yield from data_iter
        else:
            for pack in data_iter:
                # print("base pipeline reads:  " + pack.meta.doc_id)
                self.current_packs.append(pack)
                # print('now contains ', len(self.current_packs), 'packs')

                for i, processor in enumerate(self.processors):
                    for c_pack in self.current_packs:
                        in_cache = (c_pack.meta.cache_state ==
                                    processor.component_name)

                        # There is some finish bug here.
                        print('Process state: ', pack.meta.process_state)
                        can_process = (i == 0 or c_pack.meta.process_state ==
                                       self.processors[i - 1].component_name)
                        if can_process and not in_cache:
                            self.__working_component = processor.component_name
                            processor.process(c_pack)

                for c_pack in list(self.current_packs):
                    # must iterate through a copy of the original list
                    # because of the removing operation
                    if (c_pack.meta.process_state ==
                            self.processors[-1].component_name):
                        print('yielding from current pack')
                        yield c_pack
                        self.current_packs.remove(c_pack)

            # process tail instances in the whole dataset
            for c_pack in list(self.current_packs):
                start = self._processors_index[c_pack.meta.process_state] + 1
                for processor in self.processors[start:]:
                    self.__working_component = processor.component_name
                    if isinstance(processor, BaseBatchProcessor):
                        processor.process(c_pack, tail_instances=True)
                    else:
                        processor.process(c_pack)
                yield c_pack
                self.current_packs.remove(c_pack)
