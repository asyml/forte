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

import itertools
import logging
from time import time
from typing import Any, Dict, Generic, Iterator, List, Optional, Union, Tuple, \
    Deque

import yaml

from forte.common import ProcessorConfigError
from forte.common.configuration import Config
from forte.common.exception import (
    ProcessExecutionException,
    ProcessFlowException)
from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.data.base_reader import BaseReader
from forte.data.caster import Caster
from forte.data.selector import Selector, DummySelector
from forte.evaluation.base.base_evaluator import Evaluator
from forte.pipeline_component import PipelineComponent
from forte.process_job import ProcessJob
from forte.process_manager import ProcessManager, ProcessJobStatus
from forte.processors.base import BaseProcessor
from forte.processors.base.batch_processor import BaseBatchProcessor
from forte.utils import create_class_with_kwargs

logger = logging.getLogger(__name__)

__all__ = [
    "Pipeline"
]


class ProcessBuffer:
    def __init__(self, pipeline: "Pipeline", data_iter: Iterator[PackType]):
        self.__data_iter: Iterator[PackType] = data_iter
        self.__data_exhausted = False
        self.__pipeline = pipeline
        self.__process_manager: ProcessManager = pipeline._proc_mgr

    def __iter__(self):
        return self

    def __next__(self) -> ProcessJob:
        if self.__process_manager.current_queue_index == -1:
            if self.__data_exhausted:
                # Both the buffer is empty and the data input is exhausted.
                raise StopIteration
            try:
                job_pack = next(self.__data_iter)
                job = ProcessJob(job_pack, False)

                if len(self.__pipeline.evaluator_indices) > 0:
                    gold_copy = job_pack.view()
                    self.__pipeline.add_gold_packs({job.id: gold_copy})

                self.__process_manager.add_to_queue(queue_index=0, job=job)
                self.__process_manager.current_queue_index = 0
                self.__process_manager.current_processor_index = 0
                return job
            except StopIteration:
                self.__data_exhausted = True
                job = ProcessJob(None, True)
                self.__process_manager.add_to_queue(queue_index=0, job=job)
                self.__process_manager.current_queue_index = 0
                self.__process_manager.current_processor_index = 0
                return job
        else:
            q_index = self.__process_manager.current_queue_index
            u_index = self.__process_manager.unprocessed_queue_indices[q_index]
            return self.__process_manager.current_queue[u_index]


class Pipeline(Generic[PackType]):
    r"""This controls the main inference flow of the system. A pipeline is
    consisted of a set of Components (readers and processors). The data flows
    in the pipeline as data packs, and each component will use or add
    information to the data packs.
    """

    def __init__(self, resource: Optional[Resources] = None):
        self._reader: BaseReader
        self._reader_config: Optional[Config] = None

        self._components: List[PipelineComponent] = []
        self._selectors: List[Selector] = []

        self._processors_index: Dict = {'': -1}
        self._configs: List[Optional[Config]] = []

        # Will initialize at `initialize` because the processors length is
        # unknown.
        self._proc_mgr: ProcessManager = None  # type: ignore

        self.evaluator_indices: List[int] = []

        # needed for evaluator
        self._predict_to_gold: Dict[int, PackType] = {}

        if resource is None:
            self.resource = Resources()
        else:
            self.resource = resource

        self.initialized: bool = False
        self._check_type_consistency: bool = False

        # needed for time profiling of pipeline
        self._enable_profiling: bool = False
        self._profiler: List[float] = []

    def enforce_consistency(self, enforce: bool = True):
        r"""This function determines whether the pipeline will enforce
        the content expectations specified in each pipeline component. Each
        component will check whether the input pack contains the expected data
        via checking the meta-data, and throws a
        :class:`~forte.common.exception.ExpectedEntryNotFound` if the check
        fails. When this function is called with enforce is ``True``, all the
        pipeline components would check if the input datapack record matches
        with the expected types and attributes if function
        ``expected_types_and_attributes`` is implemented
        for the processor. For example, processor A requires entry type of
        ``ft.onto.base_ontology.Sentence``, and processor B would
        produce this type in the output datapack, so ``record`` function
        of processor B writes the record of this type in the datapack
        and processor A implements ``expected_types_and_attributes`` to add this
        type. Then when the pipeline runs with enforce_consistency, processor A
        would check if this type exists in the record of the output of the
        previous pipeline component.

        Args:
            enforce: A boolean of whether to enable consistency checking
                for the pipeline or not.
        """
        self._check_type_consistency = enforce

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

    def init_from_config(self, configs: List):
        r"""Initialized the pipeline (ontology and processors) from the
        given configurations.

        Args:
            configs: The configs used to initialize the pipeline.
        """

        is_first: bool = True
        for component_config in configs:
            component = create_class_with_kwargs(
                class_name=component_config['type'],
                class_args=component_config.get('kwargs', {}),
            )

            if is_first:
                if not isinstance(component, BaseReader):
                    raise ProcessorConfigError(
                        "The first component of a pipeline must be a reader.")
                self.set_reader(component, component_config.get('configs', {}))
                is_first = False
            else:
                # Can be processor, caster, or evaluator
                self.add(component, component_config.get('configs', {}))

    def set_profiling(self, enable_profiling: bool = True):
        r"""Set profiling option.

        Args:
            enable_profiling: A boolean of whether to enable profiling
                for the pipeline or not (the default is True).
        """
        self._enable_profiling = enable_profiling

    def initialize(self) -> 'Pipeline':
        # The process manager need to be assigned first.
        self._proc_mgr = ProcessManager(len(self._components))
        self._reader.initialize(self.resource, self._reader_config)
        if self._check_type_consistency:
            self.reader.enforce_consistency(enforce=True)
        else:
            self.reader.enforce_consistency(enforce=False)
        self.initialize_processors()
        self.initialized = True

        # Create profiler
        if self._enable_profiling:
            self.reader.set_profiling(True)
            self._profiler = [0.0] * len(self.components)

        return self

    def initialize_processors(self):
        for processor, config in zip(self.components, self.processor_configs):
            try:
                processor.initialize(self.resource, config)
                if self._check_type_consistency:
                    processor.enforce_consistency(enforce=True)
                else:
                    processor.enforce_consistency(enforce=False)
            except ProcessorConfigError as e:
                logging.error("Exception occur when initializing "
                              "processor %s", processor.name)
                raise e

    def set_reader(
            self, reader: BaseReader,
            config: Optional[Union[Config, Dict[str, Any]]] = None
    ) -> 'Pipeline':
        self._reader = reader
        self._reader_config = reader.make_configs(config)
        return self

    @property
    def reader(self):
        return self._reader

    @property
    def components(self):
        return self._components

    @property
    def processor_configs(self):
        return self._configs

    def add(
            self, component: PipelineComponent,
            config: Optional[Union[Config, Dict[str, Any]]] = None,
            selector: Optional[Selector] = None
    ) -> 'Pipeline':
        self._processors_index[component.name] = len(self.components)

        if isinstance(component, BaseReader):
            raise ProcessFlowException("Reader need to be set via set_reader()")

        if isinstance(component, Evaluator):
            # This will ask the job to keep a copy of the gold standard.
            self.evaluator_indices.append(len(self.components))

        self._components.append(component)
        self.processor_configs.append(component.make_configs(config))

        if selector is None:
            self._selectors.append(DummySelector())
        else:
            self._selectors.append(selector)

        return self

    def add_gold_packs(self, pack):
        r"""Add gold packs to the dictionary. This dictionary is used by the
        evaluator while calling `consume_next(...)`

        Args:
            pack (Dict): A key, value pair containing job.id -> gold_pack
                mapping
        """
        self._predict_to_gold.update(pack)

    def process(self, *args, **kwargs) -> PackType:
        r"""Alias for :meth:`process_one`.

        Args:
            args: The positional arguments used to get the initial data.
            kwargs: The keyword arguments used to get the initial data.
        """
        return self.process_one(*args, **kwargs)

    def run(self, *args, **kwargs):
        r"""Run the whole pipeline and ignore all returned DataPack. This is
        mostly used when you need to run the pipeline and do not require the
        output but rely on the side-effect. For example, if the pipeline
        writes some data to disk.

        Calling this function will automatically call the :meth:`initialize`
        at the beginning, and call the :meth:`finish` at the end.

        Args:
            args: The positional arguments used to get the initial data.
            kwargs: The keyword arguments used to get the initial data.
        """
        self.initialize()
        for _ in self.process_dataset(*args, **kwargs):
            # Process the whole dataset ignoring the return values.
            # This essentially expect the processors have side effects.
            pass
        self.finish()

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
        if not self.initialized:
            raise ProcessFlowException(
                "Please call initialize before running the pipeline")

        first_pack = []

        for p in self._reader.iter(*args, **kwargs):
            first_pack.append(p)
            break

        if len(first_pack) == 1:
            results = list(self._process_packs(iter(first_pack)))
            return results[0]
        else:
            raise ValueError("Input data source contains no packs.")

    def process_dataset(self, *args, **kwargs) -> Iterator[PackType]:
        r"""Process the documents in the data source(s) and return an
        iterator or list of DataPacks. The arguments are directly passed
        to the reader to take data from the source.
        """
        if not self.initialized:
            raise ProcessFlowException(
                "Please call initialize before running the pipeline")

        data_iter = self._reader.iter(*args, **kwargs)
        return self._process_packs(data_iter)

    def finish(self):
        """
        Call the finish method of all pipeline component. This need to be called
        explicitly to release all resources.

        Returns:

        """

        # Report time profiling of readers and processors
        if self._enable_profiling:
            out_header: str = "Pipeline Time Profile\n"
            out_reader: str = f"- Reader: {self.reader.component_name}, " + \
                f"{self.reader.time_profile} s\n"
            out_processor: str = '\n'.join([
                f"- Component [{i}]: {self.components[i].name}, {t} s"
                for i, t in enumerate(self._profiler)])
            logger.info("%s%s%s", out_header, out_reader, out_processor)

        self.reader.finish(self.resource)
        for p in self.components:
            p.finish(self.resource)

    def __update_stream_job_status(self):
        q_index = self._proc_mgr.current_queue_index
        u_index = self._proc_mgr.unprocessed_queue_indices[q_index]
        current_queue = self._proc_mgr.current_queue

        for job_i in itertools.islice(current_queue, 0, u_index + 1):
            if job_i.status == ProcessJobStatus.UNPROCESSED:
                job_i.set_status(ProcessJobStatus.PROCESSED)

    def __update_batch_job_status(self, component: BaseBatchProcessor):
        # update the status of the jobs. The jobs which were removed from
        # data_pack_pool will have status "PROCESSED" else they are "QUEUED"
        q_index = self._proc_mgr.current_queue_index
        u_index = self._proc_mgr.unprocessed_queue_indices[q_index]
        current_queue = self._proc_mgr.current_queue

        data_pool_length = len(component.batcher.data_pack_pool)

        for i, job_i in enumerate(
                itertools.islice(current_queue, 0, u_index + 1)):
            if i <= u_index - data_pool_length:
                job_i.set_status(ProcessJobStatus.PROCESSED)
            else:
                job_i.set_status(ProcessJobStatus.QUEUED)

    def __flush_batch_job_status(self):
        current_queue = self._proc_mgr.current_queue
        for job in current_queue:
            job.set_status(ProcessJobStatus.PROCESSED)

    def _process_with_component(
            self, selector: Selector, component: PipelineComponent,
            raw_job: ProcessJob):
        for pack in selector.select(raw_job.pack):
            # First, perform the component action on the pack
            try:
                if isinstance(component, Caster):
                    # Replacing the job pack with the casted version.
                    raw_job.alter_pack(component.cast(pack))
                elif isinstance(component, BaseBatchProcessor):
                    pack.set_control_component(component.name)
                    component.process(pack)
                elif isinstance(component, Evaluator):
                    pack.set_control_component(component.name)
                    component.consume_next(
                        pack, self._predict_to_gold[raw_job.id]
                    )
                elif isinstance(component, BaseProcessor):
                    # Should be BasePackProcessor:
                    # All other processor are considered to be
                    # streaming processor like this.
                    pack.set_control_component(component.name)
                    component.process(pack)
                # After the component action, make sure the entry is
                # added into the index.
                pack.add_all_remaining_entries()
            except ValueError as e:
                raise ProcessExecutionException(
                    f'Exception occurred when running '
                    f'{component.name}') from e

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
        # batch is not full according to the batcher of that processor.
        # In that case, the pipeline requests for additional jobs from the
        # reader and starts the execution loop from the beginning.
        #
        # 4) At any point, while moving to the next processor, the pipeline
        # ensures that all jobs are either in QUEUED or PROCESSED status. If
        # they are PROCESSED, they will be moved to the next queue. This design
        # ensures that at any point, while processing the job at processor `i`,
        # all the jobs in the previous queues are in QUEUED status. So whenever
        # a new job is needed, the pipeline can directly request it from the
        # reader instead of looking at previous queues for UNPROCESSED jobs.
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
        #          |___________|
        #          |___________|
        #          |___________|
        #          |___________|
        #          |_J1:QUEUED_|
        #
        # B1 needs another pack to process job J1
        #
        # After 2nd step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #          |___________|       |_______________|
        #          |___________|       |_______________|
        #          |___________|       |_______________|
        #          |___________|       |_J2:UNPROCESSED_|
        #          |___________|       |_J1:UNPROCESSED_|
        #
        # B1 processes both the packs, the jobs are moved to the next queue.
        #
        # After 3rd step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #          |___________|       |_______________|     |_______________|
        #          |___________|       |_______________|     |_______________|
        #          |___________|       |_______________|     |_______________|
        #          |___________|       |_______________|     |_______________|
        #          |___________|       |_J2:UNPROCESSED_|     |_J1:UNPROCESSED_|
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
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_J2:UNPROCESSED_|
        #        |___________|       |_______________|     |_J1:UNPROCESSED_|
        #
        #
        # After 5th step (iteration),
        #
        #           batch_size = 2                               batch_size = 2
        # Reader -> B1 (BatchProcessor) -> P1 (PackProcessor) -> B2(BatchProcessor)
        #
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|    --> Yield J1.pack and J2.pack
        #        |___________|       |_______________|     |_______________|
        #        |___________|       |_______________|     |_______________|

        if not self.initialized:
            raise ProcessFlowException(
                "Please call initialize before running the pipeline")

        buffer = ProcessBuffer(self, data_iter)

        if len(self.components) == 0:
            yield from data_iter
            # Write return here instead of using if..else to reduce indent.
            return

        while not self._proc_mgr.exhausted():
            # Take the raw job from the buffer, the job status now should
            # be UNPROCESSED.
            raw_job: ProcessJob = next(buffer)

            component_index = self._proc_mgr.current_processor_index
            component = self.components[component_index]
            selector: Selector = self._selectors[component_index]
            current_queue_index = self._proc_mgr.current_queue_index
            current_queue: Deque[ProcessJob] = self._proc_mgr.current_queue
            pipeline_length = self._proc_mgr.pipeline_length
            unprocessed_queue_indices = self._proc_mgr.unprocessed_queue_indices
            processed_queue_indices = self._proc_mgr.processed_queue_indices
            next_queue_index = current_queue_index + 1
            should_yield = next_queue_index >= pipeline_length

            if not raw_job.is_poison:

                # Start timer
                if self._enable_profiling:
                    start_time: float = time()

                self._process_with_component(selector, component, raw_job)

                # Stop timer and add to time profiler
                if self._enable_profiling:
                    self._profiler[component_index] += time() - start_time

                # Then, based on component type, handle the queue.
                if isinstance(component, BaseBatchProcessor):
                    self.__update_batch_job_status(component)
                    index = unprocessed_queue_indices[current_queue_index]

                    # Check status of all the jobs up to "index".
                    for i, job_i in enumerate(
                            itertools.islice(current_queue, 0, index + 1)):
                        if job_i.status == ProcessJobStatus.PROCESSED:
                            processed_queue_indices[current_queue_index] = i

                    # There are UNPROCESSED jobs in the queue.
                    if index < len(current_queue) - 1:
                        unprocessed_queue_indices[current_queue_index] += 1
                    elif processed_queue_indices[current_queue_index] == -1:
                        # Fetch more data from the reader to process the
                        # first job.
                        unprocessed_queue_indices[
                            current_queue_index] = len(current_queue)
                        self._proc_mgr.current_processor_index = 0
                        self._proc_mgr.current_queue_index = -1
                    else:
                        processed_queue_index = processed_queue_indices[
                            current_queue_index]
                        # Move or yield the pack.
                        c_queue = list(current_queue)
                        for job_i in c_queue[:processed_queue_index + 1]:
                            if job_i.status == ProcessJobStatus.PROCESSED:
                                if should_yield:
                                    if job_i.id in self._predict_to_gold:
                                        self._predict_to_gold.pop(job_i.id)
                                    # TODO: I don't know why these are
                                    #  marked as incompatible type by mypy.
                                    #  the same happens 3 times on every yield.
                                    #  It is observed that the pack returned
                                    #  from the `ProcessJob` is considered to
                                    #  be different from `PackType`.
                                    yield job_i.pack  # type: ignore
                                else:
                                    self._proc_mgr.add_to_queue(
                                        queue_index=next_queue_index, job=job_i)
                            else:
                                raise ProcessFlowException(
                                    f"The job status should be "
                                    f"{ProcessJobStatus.PROCESSED} "
                                    f"at this point.")
                            current_queue.popleft()

                        # Set the UNPROCESSED and PROCESSED indices.
                        unprocessed_queue_indices[
                            current_queue_index] = len(current_queue)

                        processed_queue_indices[current_queue_index] = -1

                        if should_yield:
                            self._proc_mgr.current_processor_index = 0
                            self._proc_mgr.current_queue_index = -1
                        else:
                            self._proc_mgr.current_processor_index \
                                = next_queue_index
                            self._proc_mgr.current_queue_index \
                                = next_queue_index
                # Besides Batch Processors, the other component type only
                # deal with one pack at a time, these include: PackProcessor
                # Evaluator, Caster.
                # - Move them to the next queue
                else:
                    self.__update_stream_job_status()
                    index = unprocessed_queue_indices[current_queue_index]

                    # there are UNPROCESSED jobs in the queue
                    if index < len(current_queue) - 1:
                        unprocessed_queue_indices[current_queue_index] += 1
                    else:
                        # current_queue is modified in this array
                        for job_i in list(current_queue):
                            if job_i.status == ProcessJobStatus.PROCESSED:
                                if should_yield:
                                    if job_i.id in self._predict_to_gold:
                                        self._predict_to_gold.pop(job_i.id)
                                    yield job_i.pack  # type: ignore
                                else:
                                    self._proc_mgr.add_to_queue(
                                        queue_index=next_queue_index, job=job_i)
                                current_queue.popleft()
                            else:
                                raise ProcessFlowException(
                                    f"The job status should be "
                                    f"{ProcessJobStatus.PROCESSED} "
                                    f"at this point.")

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
                            self._proc_mgr.current_processor_index = 0
                            self._proc_mgr.current_queue_index = -1

                        else:
                            self._proc_mgr.current_processor_index \
                                = next_queue_index
                            self._proc_mgr.current_queue_index \
                                = next_queue_index
            else:
                component.flush()
                self.__flush_batch_job_status()

                # current queue is modified in the loop
                for job in list(current_queue):
                    if job.status != ProcessJobStatus.PROCESSED and \
                            not job.is_poison:
                        raise ValueError("Job is neither PROCESSED nor is "
                                         "a poison. Something went wrong "
                                         "during execution.")

                    if not job.is_poison and should_yield:
                        if job.id in self._predict_to_gold:
                            self._predict_to_gold.pop(job.id)
                        yield job.pack  # type: ignore

                    elif not should_yield:
                        self._proc_mgr.add_to_queue(
                            queue_index=next_queue_index, job=job)

                    if not job.is_poison:
                        current_queue.popleft()

                if not should_yield:
                    # set next processor and queue as current
                    self._proc_mgr.current_processor_index = next_queue_index
                    self._proc_mgr.current_queue_index = next_queue_index

        self._proc_mgr.reset()

    def evaluate(self) -> Iterator[Tuple[str, Any]]:
        for i in self.evaluator_indices:
            p = self.components[i]
            assert isinstance(p, Evaluator)
            yield p.name, p.get_result()
