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
from typing import Deque, List

from forte.process_job import ProcessJob, ProcessJobStatus


class _ProcessManager:
    r"""A pipeline level manager that manages global processing information,
    such as the current running components. This is an internal class and
    should only be initialized by the system.

    Attributes:
        pipeline_length (int): The length of the current pipeline being
            executed

        _queues (List[Deque[int]]): A list of queues which hold the jobs for
            each processors. The size of this list is equal to pipeline
            length

        _current_queue_index (int): An index indicating which queue to
            read the data from. A value of -1 indicates read from the reader.

        _current_processor_index (int): An index indicating the
            processor that executes the job

        _unprocessed_queue_indices (List[int]): Each element of this list is
            the index of the first UNPROCESSED element in the corresponding
            queue. Length of this list equals the "pipeline_length".

            If unprocessed_queue_indices = [0, 2]

                - This means for the 1st queue, the first UNPROCESSED job is at
                  index-0. All elements from indices [0, len(queue[0]) ) are
                  UNPROCESSED.

                - Similarly, for the 2nd queue, the first UNPROCESSED job is at
                  index-2. All elements from indices [2, len(queue[1])) are
                  UNPROCESSED

        _processed_queue_indices (List [int]):  Each element of this list is
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

    def __init__(self, pipeline_length):
        self._pipeline_length: int = pipeline_length
        self._queues: List[Deque[ProcessJob]] = [
            deque() for _ in range(pipeline_length)]
        self._current_queue_index: int = -1
        self._current_processor_index: int = 0
        self._unprocessed_queue_indices: List[int] = [0] * pipeline_length
        self._processed_queue_indices: List[int] = [-1] * pipeline_length

    @property
    def current_processor_index(self):
        return self._current_processor_index

    @current_processor_index.setter
    def current_processor_index(self, processor_index: int):
        if processor_index >= len(self._queues):
            raise ValueError(f"{processor_index} exceeds the pipeline "
                             f"range [0, {self.pipeline_length - 1}]")
        self._current_processor_index = processor_index

    @property
    def current_queue_index(self):
        return self._current_queue_index

    @current_queue_index.setter
    def current_queue_index(self, queue_index: int):
        if queue_index >= len(self._queues):
            raise ValueError(f"{queue_index} exceeds the pipeline range "
                             f"[0, {self.pipeline_length - 1}]")
        self._current_queue_index = queue_index

    @property
    def unprocessed_queue_indices(self):
        return self._unprocessed_queue_indices

    @property
    def processed_queue_indices(self):
        return self._processed_queue_indices

    @property
    def current_queue(self):
        return self._queues[self.current_queue_index]

    @property
    def pipeline_length(self):
        return self._pipeline_length

    def add_to_queue(self, queue_index: int, job: ProcessJob):
        if queue_index > len(self._queues):
            raise ValueError(f"Queue number {queue_index} exceeds queue "
                             f"size {len(self._queues)}")
        else:
            # change the job status
            job.set_status(ProcessJobStatus.UNPROCESSED)
            self._queues[queue_index].append(job)

    def exhausted(self):
        r"""Returns True only if the last element remaining in the last queue is
         a poison pack."""

        return (len(self._queues[self.pipeline_length - 1]) == 1
                and self._queues[self.pipeline_length - 1][0].is_poison)
