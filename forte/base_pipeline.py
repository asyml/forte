import logging
from abc import abstractmethod
from typing import List, Dict, Iterator, Generic, Optional

import yaml
from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.base_pack import PackType
from forte.data.ontology import base_ontology
from forte.data.readers import BaseReader
from forte.data.selector import Selector, DummySelector
from forte.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "BasePipeline"
]


class ProcessJob:
    def __init__(self, step_num: int, pack: PackType):
        self.step_num = step_num
        self.pack = pack

    def increment(self):
        self.step_num += 1


class ProcessBuffer:

    def __init__(self, data_iter: Iterator[PackType], total_step: int):
        self.__data_iter: Iterator[PackType] = data_iter
        self.__buffer: List[ProcessJob] = []
        self.__data_exhausted = False
        self.__total_step = total_step

    def __iter__(self):
        return self

    def __next__(self) -> ProcessJob:
        """
        Create one job for the processor to process.
        Returns:

        """
        if len(self.__buffer) == 0:
            if self.__data_exhausted:
                # Both the buffer is empty and the data input is exhausted.
                raise StopIteration
            try:
                job_pack = next(self.__data_iter)
                return ProcessJob(0, job_pack)
            except StopIteration:
                self.__data_exhausted = True
                return ProcessJob(0, DataPack.make_poison())
        else:
            return self.__buffer.pop()

    def queue_process(self, job: ProcessJob):
        """
        Add a job back to the buffer to wait in the process queue. This will
        only add the job if the job step is less than the total steps (i.e. the
        job is not fully processed by all the processors)

        Args:
            job: The job contains the pack and the job step it is at.

        Returns:

        """
        job.increment()
        if job.step_num < self.__total_step:
            self.__buffer.append(job)
            return True
        else:
            return False


class BasePipeline(Generic[PackType]):
    """
        This controls the main inference flow of the system. A pipeline is
        consisted of a set of Components (readers and processors). The data
        flows in the pipeline as data packs, and each component will use or
        add information to the data packs.
    """

    def __init__(self, resource: Optional[Resources] = None):
        self._reader: BaseReader
        self._processors: List[BaseProcessor] = []
        self._selectors: List[Selector] = []

        self._processors_index: Dict = {'': -1}
        self._configs: List[Optional[HParams]] = []

        self.__working_component: str

        self._ontology = base_ontology

        if resource is None:
            self.resource = Resources()
        else:
            self.resource = resource

    def init_from_config_path(self, config_path):
        """
        Read the configs from the given path ``config_path``
        and build the pipeline with the config.

        Args:
            config_path: A string of the configuration path, which is
            is a YAML file that specify the structure and parameters of the
            pipeline.

        Returns:

        """
        configs = yaml.safe_load(open(config_path))
        self.init_from_config(configs)

    @abstractmethod
    def init_from_config(self, configs: Dict):
        """
        Initialized the pipeline (ontology and processors) from given configs

        Args:
            configs: The configs used to initialize the pipeline.

        Returns:

        """
        raise NotImplementedError

    def initialize(self):
        self._reader.initialize(resource=self.resource, configs=None)
        self.initialize_processors()

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

        if len(self.processors) > 0:
            # Indicate this as the last processor.
            self.processors[-1].set_as_last()

    def set_reader(self, reader: BaseReader):
        # reader.set_ontology(self._ontology)
        self._reader = reader

    @property
    def processors(self):
        return self._processors

    @property
    def processor_configs(self):
        return self._configs

    def add_processor(self, processor: BaseProcessor,
                      config: Optional[HParams] = None,
                      selector: Optional[Selector] = None):
        self._processors_index[processor.component_name] = len(self.processors)

        self._processors.append(processor)
        self.processor_configs.append(config)

        if selector is None:
            self._selectors.append(DummySelector())
        else:
            self._selectors.append(selector)

    def process(self, *args, **kwargs) -> PackType:
        """
        Alias for process_one.

        Args:
            args: The positional arguments used to get the initial data.
            kwargs: The keyword arguments used to get the initial data.

        Returns:

        """
        return self.process_one(*args, **kwargs)

    def run(self, *args, **kwargs):
        """
        Run the whole pipeline and ignore all returned DataPack. This is used
        when the users are relying on the side effect of the processors (e.g.
        a process that will write Packs to disk).

        Args:
            args: The positional arguments used to get the initial data.
            kwargs: The keyword arguments used to get the initial data.

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
            results = [p for p in self.process_packs(iter(first_pack))]
            return results[0]
        else:
            raise ValueError("Input data source contains no packs.")

    def process_dataset(self, *args, **kwargs) -> Iterator[PackType]:
        """
        Process the documents in the data source(s) and return an
        iterator or list of DataPacks.

        Args:
            **kwargs, which can be one or more data sources.
        """
        data_iter = self._reader.iter(*args, **kwargs)
        return self.process_packs(data_iter)

    def process_packs(
            self, data_iter: Iterator[PackType]) -> Iterator[PackType]:
        """
        Process an iterator of data packs and return the  processed ones.

        Args:
            data_iter: An iterator of the data packs.

        Returns: A list data packs.

        """
        buf = ProcessBuffer(data_iter, len(self._processors))

        if len(self.processors) == 0:
            yield from data_iter
        else:
            for job in buf:
                if not job.pack.is_poison():
                    s = self._selectors[job.step_num]
                    for c_pack in s.select(job.pack):
                        self._processors[job.step_num].process(c_pack)
                else:
                    # Pass the poison pack to the processor, so they know this
                    # is ending.
                    self._processors[job.step_num].process(job.pack)

                # Put the job back to the process queue, if not success, that
                # means this job is done processing.
                if not buf.queue_process(job):
                    done_pack: PackType = job.pack
                    if not done_pack.is_poison():
                        yield done_pack

            # # Keep a list of packs and only release it when all processors
            # # are done with them.
            # packs = []
            #
            # for pack in data_iter:
            #     packs.append(pack)
            #
            #     for i, (processor, selector) in enumerate(
            #             zip(self._processors, self._selectors)):
            #         for p in packs:
            #             for c_pack in selector.select(p):
            #                 in_cache = (c_pack.meta.cache_state ==
            #                             processor.component_name)
            #                 # TODO: can_process needs double check.
            #                 # We need to record a step here with a number
            #                 # instead of a processor component
            #                 # And we need a clean way to record whether we are
            #                 # done processing anything, the component_name
            #                 # is not reliable, especially used together with
            #                 # a selector.
            #                 can_process = (
            #                         i == 0 or c_pack.meta.process_state ==
            #                         self.processors[i - 1].component_name)
            #                 if can_process and not in_cache:
            #                     self.__working_component = \
            #                         processor.component_name
            #                     processor.process(c_pack)
            #
            #     for p in list(packs):
            #         # must iterate through a copy of the original list
            #         # because of the removing operation
            #         # TODO we'd better add a special component_name instead of
            #         #   using the previous processor. The can also cause some
            #         #   indexing problem.
            #         if (p.meta.process_state ==
            #                 self.processors[-1].component_name):
            #             yield p
            #             packs.remove(p)
            #
            # # Now the data iteration is over. We may still have some packs
            # # that are not fully processed. Now we "flush" them.
            #
            # # A special poison pack is added to the end of the data stream. It
            # # will not be processed by any of the processors, but it will tell
            # # the processors that the stream ends.
            # for p in list(packs) + [BasePack.get_poison()]:
            #     # TODO double check starts
            #     start = self._processors_index[p.meta.process_state] + 1
            #     for processor, selector in zip(self._processors[start:],
            #                                    self._selectors):
            #         self.__working_component = processor.component_name
            #
            #         if not p.is_poison():
            #             for c_pack in selector.select(p):
            #                 processor.process(c_pack)
            #         else:
            #             processor.process(p)
            #
            #     # And we certainly won't return the poison pack.
            #     if not p.is_poison():
            #         yield p
            #     packs.remove(p)
