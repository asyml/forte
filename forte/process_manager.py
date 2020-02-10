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

from collections import deque
from enum import Enum
from typing import Deque, List, Optional

__all__ = [
    "ProcessManager",
    "ProcessJobStatus"
]

ProcessJobStatus = Enum("ProcessJobStatus", "UNPROCESSED QUEUED PROCESSED")


class ProcessManager:
    r"""A pipeline level manager that manages global processing information,
    such as the current running components."""

    # Note: hiding the real class creation and attributes here allow us to
    # create a singleton class. A singleton ProcessManager should be sufficient
    # when we are not dealing with multi-process.

    class __ProcessManager:
        def __init__(self):
            self.current_component: str = '__default__'
            self.pipeline_length: int
            self.queues: List[Deque[int]]
            self.current_queue_index: int
            self.current_processor_index: int
            self.unprocessed_queue_indices: List[int]
            self.processed_queue_indices: List[int]

    instance: Optional[__ProcessManager] = None

    def __init__(self):
        if not ProcessManager.instance:
            ProcessManager.instance = ProcessManager.__ProcessManager()

    # pylint: disable=attribute-defined-outside-init
    def initialize_queues(self, pipeline_length: int):
        r"""Initialize the queues and the related state variables to execute
        the pipeline. Following variables are defined in this method

        - pipeline_length (int): The length of the current pipeline being
            executed

        - queues (List[Deque[int]]): A list of queues which hold the jobs for
            each processors. The size of this list is equal to pipeline
            length

        - current_queue_index (int): An index indicating which queue to
            read the data from. A value of -1 indicates read from the reader.

        - current_processor_index (int): An index indicating the
            processor that executes the job

        - unprocessed_queue_indices (List[int]): Each element of this list is
            the index of the first UNPROCESSED element in the corresponding
            queue. Length of this list equals the "pipeline_length".

            If unprocessed_queue_indices = [0, 2]

                - This means for the 1st queue, the first UNPROCESSED job is at
                  index-0. All elements from indices [0, len(queue[0]) ) are
                  UNPROCESSED.

                - Similarly, for the 2nd queue, the first UNPROCESSED job is at
                  index-2. All elements from indices [2, len(queue[1])) are
                  UNPROCESSED

        - processed_queue_indices (List [int]):  Each element of this list is
            the index of the last PROCESSED element in the corresponding queue.
            Length of this list equals the "pipeline_length".

            If processed_queue_indices = [0, 2]

                - This means for the 1st queue, the last PROCESSED job is at
                  index-0. Only the first element in queue[0] is PROCESSED

                - Similarly, for the 2nd queue, the last PROCESSED job is at
                  index-2. All elements from indices [0, 2] are PROCESSED

        Args:
            pipeline_length (int): The length of the current pipeline being
                executed

        """
        if self.instance is not None:
            self.instance.pipeline_length = pipeline_length
            self.instance.queues = [deque() for _ in range(pipeline_length)]
            self.instance.current_queue_index = -1
            self.instance.current_processor_index = 0
            self.instance.unprocessed_queue_indices = [0] * pipeline_length
            self.instance.processed_queue_indices = [-1] * pipeline_length
        else:
            raise AttributeError("The process manager is not initialized.")

    def set_current_pipeline(self, pipeline):
        r"""Set the current pipeline.

        Args:
            pipeline (Pipeline): The current pipeline to be executed
        """
        # In the current design, :class:`ProcessManager` holds all the state
        # variables related to a pipeline. As an extension,
        # :class:`ProcessManager` also holds the reference to the current
        # pipeline being executed. This will be useful in places where any class
        # needs to access the pipeline through :class:`ProcessManager`. As a
        # concrete example, take a look at
        # :class:`forte.base_pipeline.ProcessBuffer`
        if self.instance is not None:
            self.instance.current_pipeline = pipeline
        else:
            raise AttributeError('The process manager is not initialized.')

    def set_current_component(self, component_name: str):
        r"""Set the current component

        Args:
             component_name (str): Current component name
        """
        if self.instance is not None:
            self.instance.current_component = component_name
        else:
            raise AttributeError('The process manager is not initialized.')

    def set_current_processor_index(self, processor_index: int):
        if self.instance is not None:
            if processor_index >= len(self.instance.queues):
                raise ValueError(f"{processor_index} exceeds the pipeline "
                                 f"range [0, {self.pipeline_length - 1}]")
            self.instance.current_processor_index = processor_index
        else:
            raise AttributeError("The process manager is not initialized.")

    def set_current_queue_index(self, queue_index: int):
        if self.instance is not None:
            if queue_index >= len(self.instance.queues):
                raise ValueError(f"{queue_index} exceeds the pipeline range "
                                 f"[0, {self.pipeline_length - 1}]")
            self.instance.current_queue_index = queue_index
        else:
            raise AttributeError("The process manager is not initialized.")

    @property
    def current_pipeline(self):
        if self.instance is not None:
            return self.instance.current_pipeline
        else:
            raise AttributeError("The process manager is not initialized.")

    @property
    def current_processor_index(self):
        if self.instance is not None:
            return self.instance.current_processor_index
        else:
            raise AttributeError("The process manager is not initialized.")

    @property
    def current_queue_index(self):
        if self.instance is not None:
            return self.instance.current_queue_index
        else:
            raise AttributeError("The process manager is not initialized.")

    @property
    def unprocessed_queue_indices(self):
        if self.instance is not None:
            return self.instance.unprocessed_queue_indices
        else:
            raise AttributeError("The process manager is not initialized.")

    @property
    def processed_queue_indices(self):
        if self.instance is not None:
            return self.instance.processed_queue_indices
        else:
            raise AttributeError("The process manager is not initialized.")

    @property
    def current_queue(self):
        if self.instance is not None:
            return self.instance.queues[self.current_queue_index]
        else:
            raise AttributeError("The process manager is not initialized.")

    @property
    def pipeline_length(self):
        if self.instance is not None:
            return self.instance.pipeline_length
        else:
            raise AttributeError("The process manager is not initialized.")

    def add_to_queue(self, queue_index, job):
        if self.instance is not None:
            if queue_index > len(self.instance.queues):
                raise ValueError(f"Queue number {queue_index} exceeds queue "
                                 f"size {len(self.instance.queues)}")
            else:
                # change the job status
                job.set_status(ProcessJobStatus.UNPROCESSED)
                self.instance.queues[queue_index].append(job)
        else:
            raise AttributeError("The process manager is not initialized.")

    def exhausted(self):
        r"""Returns True only if the last element remaining in the last queue is
         a poison pack."""

        return len(self.instance.queues[self.pipeline_length - 1]) == 1 and \
               self.instance.queues[self.pipeline_length - 1][0].is_poison

    @property
    def component(self):
        return self.instance.current_component
