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
"""
Base class for Pipeline module.
"""

import logging
import itertools

from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Optional, Union

import yaml

from texar.torch import HParams

from forte.common.evaluation import Evaluator
from forte.common.resources import Resources
from forte.data.readers.base_reader import BaseReader
from forte.data.base_pack import PackType
from forte.data.selector import Selector, DummySelector
from forte.processors.base.base_processor import BaseProcessor
from forte.processors.base.batch_processor import BaseBatchProcessor
from forte.process_manager import ProcessManager, ProcessJobStatus


logger = logging.getLogger(__name__)

__all__ = [
    "BasePipeline"
]

process_manager = ProcessManager()


class ProcessJob:

    def __init__(self, pack: Optional[PackType], is_poison: bool):
        self.__pack: Optional[PackType] = pack
        self.__is_poison: bool = is_poison
        self.__status = ProcessJobStatus.UNPROCESSED

    def set_status(self, status):
        self.__status = status

    @property
    def pack(self) -> PackType:
        if self.__pack is None:
            raise ValueError("This job do not have a valid pack.")
        return self.__pack

    @property
    def is_poison(self) -> bool:
        return self.__is_poison

    @property
    def status(self):
        return self.__status


class ProcessBuffer:

    def __init__(self, data_iter: Iterator[PackType]):
        self.__data_iter: Iterator[PackType] = data_iter
        self.__data_exhausted = False

    def __iter__(self):
        return self

    def __next__(self) -> ProcessJob:
        if process_manager.current_queue_index == -1:
            if self.__data_exhausted:
                # Both the buffer is empty and the data input is exhausted.
                raise StopIteration
            try:
                job_pack = next(self.__data_iter)
                job = ProcessJob(job_pack, False)
                process_manager.add_to_queue(queue_index=0, job=job)
                process_manager.set_current_queue_index(queue_index=0)
                process_manager.set_current_processor_index(processor_index=0)
                return job
            except StopIteration:
                self.__data_exhausted = True
                job = ProcessJob(None, True)
                process_manager.add_to_queue(queue_index=0, job=job)
                process_manager.set_current_queue_index(queue_index=0)
                process_manager.set_current_processor_index(processor_index=0)
                return job
        else:
            q_index = process_manager.current_queue_index
            u_index = process_manager.unprocessed_queue_indices[q_index]
            return process_manager.current_queue[u_index]


class BasePipeline(Generic[PackType]):
    r"""This controls the main inference flow of the system. A pipeline is
    consisted of a set of Components (readers and processors). The data flows
    in the pipeline as data packs, and each component will use or add
    information to the data packs.
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
        r"""Read the configurations from the given path ``config_path``
        and build the pipeline with the config.

        Args:
            config_path: A string of the configuration path, which is
                is a YAML file that specify the structure and parameters of the
                pipeline.
        """
        configs = yaml.safe_load(open(config_path))
        self.init_from_config(configs)

    @abstractmethod
    def init_from_config(self, configs: Dict):
        r"""Initialized the pipeline (ontology and processors) from the
        given configurations.

        Args:
            configs: The configs used to initialize the pipeline.
        """
        raise NotImplementedError

    def initialize(self):
        self._reader.initialize(self.resource, self._reader_config)
        self.initialize_processors()
        process_manager.initialize_queues(pipeline_length=len(self._processors))

        if self._evaluator:
            self._evaluator.initialize(self.resource, self._evaluator_config)

    def initialize_processors(self):
        for processor, config in zip(self.processors, self.processor_configs):
            processor.initialize(self.resource, config)

    def set_reader(self, reader: BaseReader,
                   config: Optional[Union[HParams, Dict[str, Any]]] = None):
        self._reader = reader

        if config is None:
            config = reader.default_configs()
        config = HParams(config, reader.default_configs())

        self._reader_config = config

    @property
    def processors(self):
        return self._processors

    @property
    def processor_configs(self):
        return self._configs

    def add_processor(self, processor: BaseProcessor,
                      config: Optional[Union[HParams, Dict[str, Any]]] = None,
                      selector: Optional[Selector] = None):
        self._processors_index[processor.component_name] = len(self.processors)

        self._processors.append(processor)

        if config is None:
            config = processor.default_configs()
        config = HParams(config, processor.default_configs())

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
        r"""Alias for :meth:`process_one`.

        Args:
            args: The positional arguments used to get the initial data.
            kwargs: The keyword arguments used to get the initial data.
        """
        return self.process_one(*args, **kwargs)

    def run(self, *args, **kwargs):
        r"""Run the whole pipeline and ignore all returned DataPack. This is
        used when the users are relying on the side effect of the processors
        (e.g. a process that will write Packs to disk).

        Args:
            args: The positional arguments used to get the initial data.
            kwargs: The keyword arguments used to get the initial data.
        """
        for _ in self.process_dataset(*args, **kwargs):
            # Process the whole dataset ignoring the return values.
            # This essentially expect the processors have side effects.
            pass

    def process_one(self, *args, **kwargs) -> PackType:
        r"""Process one single data pack. This is done by only reading and
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
        r"""Process the documents in the data source(s) and return an
        iterator or list of DataPacks. The arguments are directly passed
        to the reader to take data from the source.
        """
        # TODO: This is a generator, but the name may be confusing since the
        #  user might expect this function will do all the processing. If
        #  this is called like `process_dataset(args)` instead of
        #  `for p in process_dataset(args)`, this will have no effect.

        # try:
        data_iter = self._reader.iter(*args, **kwargs)
        return self._process_packs(data_iter)

    def _process_packs(
            self, data_iter: Iterator[PackType]) -> Iterator[PackType]:
        r"""Process the packs received from the reader by the running through
        the pipeline.

        Args:
             data_iter (iterator): Iterator yielding jobs that contain packs

        Returns:
            Yields packs that are processed by the pipeline.
        """

        # pylint: disable=line-too-long

        # Here is the logic for the execution of the pipeline.

        # The basic idea is to yield a pack as soon as it gets processed by all
        # the processors instead of waiting for later jobs to get processed.

        # 1) A job can be in three status
        #  - UNPROCESSED
        #  - QUEUED
        #  - PROCESSED
        #
        # 2) Each processor maintains a queue to hold the jobs to be executed
        # next.
        #
        # 3) In case of a BatchProcessor, a job enters into QUEUED status if the
        # job does not satisfy the `batch_size` requirement of that processor.
        # In that case, the pipeline requests for additional jobs from the reader
        # and starts the execution loop from the beginning.
        #
        # 4) At any point, while moving to the next processor, the pipeline
        # ensures that all jobs are either in QUEUED or PROCESSED status. If they
        # are PROCESSED, they will be moved to the next queue. This design ensures
        # that at any point, while processing the job at processor `i`, all the
        # jobs in the previous queues are in QUEUED status. So whenever a new job
        # is needed, the pipeline can directly request it from the reader instead
        # of looking at previous queues for UNPROCESSED jobs.
        #
        # 5) When a processor receives a poison pack, it flushes all the
        # remaining batches in its memory (this actually has no effect in
        # PackProcessors) and moves the jobs including the poison pack to the
        # next queue. If there is no next processor, the packs are yield.
        #
        # 6) The loop terminates when the last queue contains only a poison pack
        #
        # Here is the sample pipeline and its execution
        #
        # Assume 1 pack corresponds to a batch of size 1
        #
        # After 1st step (iteration), reading from the reader,
        #
        #            batch_size = 2                               batch_size = 2
        #  Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #          |______________|
        #          |______________|
        #          |______________|
        #          |______________|
        #          |_<J1>: QUEUED_|
        #
        # B1 needs another pack to process job J1
        #
        # After 2nd step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #          |______________|       |__________________|
        #          |______________|       |__________________|
        #          |______________|       |__________________|
        #          |______________|       |_<J2>:UNPROCESSED_|
        #          |______________|       |_<J1>:UNPROCESSED_|
        #
        # B1 processes both the packs, the jobs are moved to the next queue.
        #
        # After 3rd step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #          |______________|       |__________________|     |__________________|
        #          |______________|       |__________________|     |__________________|
        #          |______________|       |__________________|     |__________________|
        #          |______________|       |__________________|     |__________________|
        #          |______________|       |_<J2>:UNPROCESSED_|     |_<J1>:UNPROCESSED_|
        #
        # P1 processes the first job. However, there exists one UNPROCESSED job
        # J2 in the queue. Pipeline first processes this job before moving to the
        # next processor
        #
        # After 4th step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #        |______________|       |__________________|     |__________________|
        #        |______________|       |__________________|     |__________________|
        #        |______________|       |__________________|     |__________________|
        #        |______________|       |__________________|     |_<J2>:UNPROCESSED_|
        #        |______________|       |__________________|     |_<J1>:UNPROCESSED_|
        #
        #
        # After 5th step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #        |______________|       |__________________|     |__________________|
        #        |______________|       |__________________|     |__________________|
        #        |______________|       |__________________|     |__________________|    --> Yield J1.pack and J2.pack
        #        |______________|       |__________________|     |__________________|
        #        |______________|       |__________________|     |__________________|

        buffer = ProcessBuffer(data_iter)

        if len(self.processors) == 0:
            yield from data_iter

        else:
            while not process_manager.exhausted():

                # job has to be the first UNPROCESSED element
                # the status of the job now is UNPROCESSED
                unprocessed_job = next(buffer)

                processor_index = process_manager.current_processor_index
                processor = self.processors[processor_index]
                selector = self._selectors[processor_index]
                current_queue_index = process_manager.current_queue_index
                current_queue = process_manager.current_queue
                pipeline_length = process_manager.pipeline_length
                unprocessed_queue_indices = \
                    process_manager.unprocessed_queue_indices
                processed_queue_indices = \
                    process_manager.processed_queue_indices
                next_queue_index = current_queue_index + 1
                should_yield = next_queue_index >= pipeline_length

                if not unprocessed_job.is_poison:

                    for pack in selector.select(unprocessed_job.pack):

                        processor.process(pack)

                        if isinstance(processor, BaseBatchProcessor):

                            index = \
                                unprocessed_queue_indices[current_queue_index]

                            # check status of all the jobs up to "index"
                            for i, job_i in enumerate(
                                    itertools.islice(current_queue, 0,
                                                     index + 1)):

                                if job_i.status == ProcessJobStatus.PROCESSED:
                                    processed_queue_indices[
                                        current_queue_index] = i

                            # there are UNPROCESSED jobs in the queue
                            if index < len(current_queue) - 1:
                                unprocessed_queue_indices[current_queue_index] \
                                    += 1

                            # Fetch more data from the reader to process the
                            # first job
                            elif (processed_queue_indices[current_queue_index]
                                  == -1):

                                unprocessed_queue_indices[current_queue_index] \
                                    = len(current_queue)

                                process_manager.set_current_processor_index(
                                    processor_index=0)

                                process_manager.set_current_queue_index(
                                    queue_index=-1)

                            else:
                                processed_queue_index = \
                                    processed_queue_indices[current_queue_index]

                                # move or yield the pack
                                c_queue = list(current_queue)
                                for job_i in \
                                        c_queue[:processed_queue_index + 1]:

                                    if should_yield:
                                        if self._evaluator:
                                            self._evaluator.consume_next(
                                                job_i.pack, job_i.pack)
                                        yield job_i.pack

                                    else:
                                        process_manager.add_to_queue(
                                            queue_index=next_queue_index,
                                            job=job_i)

                                    current_queue.popleft()

                                # set the UNPROCESSED and PROCESSED indices
                                unprocessed_queue_indices[current_queue_index] \
                                    = len(current_queue)

                                processed_queue_indices[current_queue_index] \
                                    = -1

                                if should_yield:
                                    process_manager.set_current_processor_index(
                                        processor_index=0)

                                    process_manager.set_current_queue_index(
                                        queue_index=-1)
                                else:
                                    process_manager.set_current_processor_index(
                                        processor_index=next_queue_index)
                                    process_manager.set_current_queue_index(
                                        queue_index=next_queue_index)

                        # For PackProcessor
                        # - Process all the packs in the queue and move them to
                        # the next queue
                        else:

                            index = \
                                unprocessed_queue_indices[current_queue_index]

                            # there are UNPROCESSED jobs in the queue
                            if index < len(current_queue) - 1:
                                unprocessed_queue_indices[current_queue_index] \
                                    += 1

                            else:
                                # current_queue is modified in this array
                                for job_i in list(current_queue):

                                    if should_yield:
                                        if self._evaluator:
                                            self._evaluator.consume_next(
                                                job_i.pack, job_i.pack)
                                        yield job_i.pack

                                    else:
                                        process_manager.add_to_queue(
                                            queue_index=next_queue_index,
                                            job=job_i)

                                    current_queue.popleft()

                                # set the UNPROCESSED index
                                # we do not use "processed_queue_indices" as the
                                # jobs get PROCESSED whenever they are passed
                                # into a PackProcessor
                                unprocessed_queue_indices[current_queue_index] \
                                    = len(current_queue)

                                # update the current queue and processor only
                                # when all the jobs are processed in the current
                                # queue
                                if should_yield:
                                    process_manager.set_current_processor_index(
                                        processor_index=0)

                                    process_manager.set_current_queue_index(
                                        queue_index=-1)

                                else:
                                    process_manager.set_current_processor_index(
                                        processor_index=next_queue_index)

                                    process_manager.set_current_queue_index(
                                        queue_index=next_queue_index)

                else:

                    processor.flush()

                    # current queue is modified in the loop
                    for job in list(current_queue):

                        if job.status != ProcessJobStatus.PROCESSED and \
                                not job.is_poison:
                            raise ValueError("Job is neither PROCESSED nor is "
                                             "a poison. Something went wrong "
                                             "during execution.")

                        if not job.is_poison and should_yield:
                            if self._evaluator:
                                self._evaluator.consume_next(job.pack, job.pack)
                            yield job.pack

                        elif not should_yield:
                            process_manager.add_to_queue(
                                queue_index=next_queue_index, job=job)

                        if not job.is_poison:
                            current_queue.popleft()

                    if not should_yield:
                        # set next processor and queue as current
                        process_manager.set_current_processor_index(
                            processor_index=next_queue_index)

                        process_manager.set_current_queue_index(
                            queue_index=next_queue_index)

    def evaluate(self):
        if self._evaluator:
            return self._evaluator.get_result()
        else:
            raise ValueError("Pipeline has no evaluator. "
                             "Cannot evaluate the results.")
