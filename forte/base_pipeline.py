# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from abc import abstractmethod
from typing import List, Dict, Iterator, Generic, Optional

import yaml
from texar.torch import HParams

from forte.common import Evaluator, Resources
from forte.data.base_pack import PackType
from forte.data.readers import BaseReader
from forte.data.selector import Selector, DummySelector
from forte.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "BasePipeline"
]


class ProcessJob:
    def __init__(self, step_num: int, pack: Optional[PackType],
                 is_poison: bool):
        self.__step_num: int = step_num
        self.__pack: Optional[PackType] = pack
        self.__is_poison: bool = is_poison

    def increment(self):
        self.__step_num += 1

    @property
    def step_num(self) -> int:
        return self.__step_num

    @property
    def pack(self) -> PackType:
        if self.__pack is None:
            raise ValueError("This job do not have a valid pack.")
        return self.__pack

    @property
    def is_poison(self) -> bool:
        return self.__is_poison


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
                return ProcessJob(0, job_pack, False)
            except StopIteration:
                self.__data_exhausted = True
                return ProcessJob(0, None, True)
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
        self._reader_config: Optional[HParams]

        self._processors: List[BaseProcessor] = []
        self._selectors: List[Selector] = []

        self._processors_index: Dict = {'': -1}
        self._configs: List[Optional[HParams]] = []

        self._evaluator: Optional[Evaluator] = None
        self._evaluator_config: Optional[HParams] = None

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
        self._reader.initialize(self.resource, self._reader_config)
        self.initialize_processors()
        if self._evaluator:
            self._evaluator.initialize(self.resource, self._evaluator_config)

    def initialize_processors(self):
        for processor, config in zip(self.processors, self.processor_configs):
            processor.initialize(self.resource, config)

    def set_reader(self, reader: BaseReader, config: Optional[HParams] = None):
        self._reader = reader
        self._reader_config = config

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

    def set_evaluator(self, evaluator: Evaluator,
                      config: Optional[HParams] = None):

        if not isinstance(evaluator, Evaluator):
            raise ValueError(f"Evaluator should be an instance of Evaluator. "
                             f"Got {type(evaluator)}")

        self._evaluator = evaluator
        self._evaluator_config = config

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
            results = [p for p in self._process_packs(iter(first_pack))]
            return results[0]
        else:
            raise ValueError("Input data source contains no packs.")

    def process_dataset(self, *args, **kwargs) -> Iterator[PackType]:
        """
        Process the documents in the data source(s) and return an
        iterator or list of DataPacks. The arguments are directly passed
        to the reader to take data from the source.

        Args:
            *args:
            **kwargs:
        """
        # TODO: This is a generator, but the name may be confusing since the
        #  user might expect this function will do all the processing, if
        #  this is called like `process_dataset(args)` instead of
        #  `for p in process_dataset(args)`, this will have no effect.

        # try:
        data_iter = self._reader.iter(*args, **kwargs)
        return self._process_packs(data_iter)

    def _process_packs(
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
                if not job.is_poison:
                    s = self._selectors[job.step_num]
                    for c_pack in s.select(job.pack):
                        self._processors[job.step_num].process(c_pack)
                else:
                    # Pass the poison pack to the processor, so they know this
                    # is ending.
                    self._processors[job.step_num].flush()

                # Put the job back to the process queue, if not success, that
                # means this job is done processing.
                if not buf.queue_process(job):
                    if not job.is_poison:
                        if self._evaluator:
                            self._evaluator.consume_next(job.pack, job.pack)
                        yield job.pack
                    else:
                        self._reader.finish(self.resource)
                        for processor in self.processors:
                            processor.finish(self.resource)

    def evaluate(self):
        if self._evaluator:
            return self._evaluator.get_result()
        else:
            raise ValueError("Pipeline has no evaluator. "
                             "Cannot evaluate the results.")
