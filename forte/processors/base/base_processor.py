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
The base class of processors
"""
from abc import abstractmethod, ABC
from typing import Optional

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.data.selector import DummySelector
from forte.process_manager import ProcessManager
from forte.utils import get_full_module_name
from forte.pipeline_component import PipelineComponent

__all__ = [
    "BaseProcessor",
]

process_manager = ProcessManager()


class BaseProcessor(PipelineComponent[PackType], ABC):
    r"""The basic processor class. To be inherited by all kinds of processors
    such as trainer, predictor and evaluator.
    """

    def __init__(self):
        self.component_name = get_full_module_name(self)
        self.selector = DummySelector()

    def initialize(self, resource: Resources, configs: Optional[HParams]):
        r"""The pipeline will call the initialize method at the start of a
        processing. The processor will be initialized with ``configs``,
        and register global resources into ``resource``. The implementation
        should set up the states of the processor.

        Args:
            resource: A global resource register. User can register
                shareable resources here, for example, the vocabulary.
            configs: The configuration passed in to set up this processor.
        """
        pass

    def process(self, input_pack: PackType):
        # Set the component for recording purpose.
        process_manager.set_current_component(self.component_name)
        self._process(input_pack)

    @abstractmethod
    def _process(self, input_pack: PackType):
        r"""The main function of the processor should be implemented here. The
        implementation of this function should process the ``input_pack``, and
        conduct operations such as adding entries into the pack, or produce
        some side-effect such as writing data into the disk.

        Args:
            input_pack:
        """
        raise NotImplementedError

    def flush(self):
        r"""Indicate that there will be no more packs to be passed in.
        """
        pass

    @staticmethod
    def default_hparams():
        r"""This defines a basic Hparams structure.
        """
        return {
            'selector': {
                'type': 'forte.data.selector.DummySelector',
                'args': None,
                'kwargs': {}
            }
        }
