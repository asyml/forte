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
Base class for processors.
"""

import itertools

from abc import abstractmethod, ABC
from typing import Any, Dict

from forte.data.base_pack import PackType
from forte.data.selector import DummySelector
from forte.utils.utils import get_full_module_name
from forte.pipeline_component import PipelineComponent
from forte.process_manager import ProcessManager, ProcessJobStatus

__all__ = [
    "BaseProcessor",
]

process_manager = ProcessManager()


class BaseProcessor(PipelineComponent[PackType], ABC):
    r"""Base class inherited by all kinds of processors such as trainer,
    predictor and evaluator.
    """

    def __init__(self):
        self.component_name = get_full_module_name(self)
        self.selector = DummySelector()

    def process(self, input_pack: PackType):
        # Set the component for recording purpose.
        process_manager.set_current_component(self.component_name)
        self._process(input_pack)

        # Change status for pack processors
        q_index = process_manager.current_queue_index
        u_index = process_manager.unprocessed_queue_indices[q_index]
        current_queue = process_manager.current_queue

        for job_i in itertools.islice(current_queue, 0, u_index + 1):
            if job_i.status == ProcessJobStatus.UNPROCESSED:
                job_i.set_status(ProcessJobStatus.PROCESSED)

    @abstractmethod
    def _process(self, input_pack: PackType):
        r"""The main function of the processor. The implementation should
        process the ``input_pack``, and conduct operations such as adding
        entries into the pack, or produce some side-effect such as writing
        data into the disk.

        Args:
            input_pack: The input datapack.
        """
        raise NotImplementedError

    def flush(self):
        r"""Indicate that there will be no more packs to be passed in.
        """
        pass

    @staticmethod
    def default_configs() -> Dict[str, Any]:
        r"""Returns a `dict` of configurations of the processor with default
        values. Used to replace the missing values of input ``configs`` during
        pipeline construction.
        """
        return {
            'selector': {
                'type': 'forte.data.selector.DummySelector',
                'args': None,
                'kwargs': {}
            }
        }
