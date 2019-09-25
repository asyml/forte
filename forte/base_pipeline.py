import logging
from abc import abstractmethod
from typing import List, Dict, Iterator, Generic, Optional, Union

import yaml
from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.data.base_pack import BasePack
from forte.data.ontology import base_ontology
from forte.data.readers import BaseReader
from forte.data.selector import Selector, DummySelector
from forte.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "BasePipeline"
]


class BasePipeline(Generic[PackType]):
    """
    The pipeline consists of a list of predictors.
    """

    def __init__(self, resource: Optional[Resources] = None):
        self._reader: BaseReader
        self._processors: List[BaseProcessor] = []
        self._selectors: List[Selector] = []

        self._processors_index: Dict = {'': -1}
        self._configs: List[Optional[HParams]] = []

        self.__working_component: str

        self._ontology = base_ontology
        self.topology = None
        # self.current_packs = []

        if resource is None:
            self.resource = Resources()
        else:
            self.resource = resource

    def init_from_config_path(self, config_path):
        """
        Read the configs from the given path ``config_path``
        and build the pipeline with the config.

        :param config_path: A string of the configuration path. The config_path
        is a YAML file that specify the structure and parameters of the
        processor.
        :return:
        """
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
            processor.initialize(self.resource, config)
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

    def add_processor(
            self, processor: BaseProcessor,
            config: Optional[HParams] = None,
            selector: Optional[Selector] = None,
    ):
        if self._ontology:
            processor.set_ontology(self._ontology)
        self._processors_index[processor.component_name] = len(self.processors)

        self._processors.append(processor)
        self.processor_configs.append(config)

        if selector is None:
            self._selectors.append(DummySelector())

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
            # Keep a list of packs and only release it when all processors
            # are done with them.
            packs = []
            for pack in data_iter:
                packs.append(pack)

                for i, (processor, selector) in enumerate(
                        zip(self._processors, self._selectors)):
                    for p in packs:
                        for c_pack in selector.select(p):
                            in_cache = (c_pack.meta.cache_state ==
                                        processor.component_name)
                            # TODO: can_process needs double check.
                            # We need to record a step here with a number
                            # instead of a processor component
                            # And we need a clean way to record whether we are
                            # done processing anything, the component_name
                            # is not reliable, especially used together with
                            # a selector.
                            can_process = (
                                    i == 0 or c_pack.meta.process_state ==
                                    self.processors[i - 1].component_name)
                            if can_process and not in_cache:
                                self.__working_component = \
                                    processor.component_name
                                processor.process(c_pack)

                for p in list(packs):
                    # must iterate through a copy of the original list
                    # because of the removing operation
                    # TODO we'd better add a special component_name instead of
                    # using the previous processor. The can also cause some
                    # indexing problem.
                    if (p.meta.process_state ==
                            self.processors[-1].component_name):
                        yield p
                        packs.remove(p)

            # Now the data iteration is over. We may still have some packs
            # that are not fully processed. Now we "flush" them.

            # A special poison pack is added to the end of the data stream. It
            # will not be processed by any of the processors, but it will tell
            # the processors that the stream ends.
            for c_pack in list(packs) + [BasePack.get_poison()]:
                # TODO double check starts
                start = self._processors_index[c_pack.meta.process_state] + 1
                for processor, selector in zip(self._processors[start:],
                                               self._selectors):
                    self.__working_component = processor.component_name
                    for p in c_pack:
                        processor.process(p)

                # And we certainly won't return the poison pack.
                if not c_pack.is_poison():
                    yield c_pack
                packs.remove(c_pack)
