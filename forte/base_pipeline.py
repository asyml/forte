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

    def __init__(
            self,
            data_iter: Iterator[PackType],
            total_step: int
    ):
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
        self.__reader: BaseReader
        self.__reader_config: Optional[HParams] = None

        self.__processors: List[BaseProcessor] = []
        self.__selectors: List[Selector] = []

        self.__processors_index: Dict = {'': -1}
        self.__configs: List[Optional[HParams]] = []

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
        # self._reader.initialize()
        self.initialize_components()

    def set_ontology(self, ontology):
        self._ontology = ontology
        for processor in self.processors:
            processor.set_ontology(self._ontology)

    def initialize_components(self):
        self.__reader.initialize(self.resource, self.__reader_config)

        for component, config in zip(self.__processors, self.__configs):
            component.initialize(self.resource, config)
            if isinstance(component, BaseProcessor):
                component.set_input_info()
                component.set_output_info()
        # Indicate this as the last processor.
        self.processors[-1].set_as_last()

    def set_reader(self, reader: BaseReader, config: Optional[HParams] = None):
        reader.set_ontology(self._ontology)
        self.__reader = reader
        self.__reader_config = config

    @property
    def processors(self):
        return self.__processors

    @property
    def processor_configs(self):
        return self.__configs

    def add_processor(
            self,
            processor: BaseProcessor,
            config: Optional[HParams] = None,
            selector: Optional[Selector] = None,
    ):
        self.__processors_index[processor.component_name] = len(self.processors)
        self.__processors.append(processor)
        self.__configs.append(config)

        if selector is None:
            self.__selectors.append(DummySelector())

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
        for p in self.__reader.iter(*args, **kwargs):
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
        data_iter = self.__reader.iter(*args, **kwargs)
        return self.process_packs(data_iter)

    def process_packs(
            self, data_iter: Iterator[PackType]
    ) -> Iterator[PackType]:
        """
        Process an iterator of data packs and return the  processed ones.

        Args:
            data_iter: An iterator of the data packs.

        Returns: A list data packs.

        """
        buf = ProcessBuffer(data_iter, len(self.__processors))

        if len(self.processors) == 0:
            yield from data_iter
        else:
            for job in buf:
                if not job.pack.is_poison():
                    s = self.__selectors[job.step_num]
                    for c_pack in s.select(job.pack):
                        self.__processors[job.step_num].process(c_pack)
                else:
                    # Pass the poison pack to the processor, so they know this
                    # is ending.
                    self.__processors[job.step_num].process(job.pack)

                # Put the job back to the process queue, if not success, that
                # means this job is done processing.
                if not buf.queue_process(job):
                    done_pack: PackType = job.pack
                    if not done_pack.is_poison():
                        yield done_pack
