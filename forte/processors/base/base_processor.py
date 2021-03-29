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

from abc import abstractmethod, ABC
from typing import Any, Dict, List

from forte.common import ExpectedRecordNotFound
from forte.data.base_pack import PackType
from forte.data.selector import DummySelector
from forte.pipeline_component import PipelineComponent

__all__ = [
    "BaseProcessor",
]


class BaseProcessor(PipelineComponent[PackType], ABC):
    r"""Base class inherited by all kinds of processors such as trainer,
    predictor and evaluator.
    """

    def __init__(self):
        super().__init__()
        self.selector = DummySelector()

    def record(self, record_meta: Dict):
        r"""Method to add output type record of current processor
        to :attr:`forte.data.data_pack.Meta.record`

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        pass

    @classmethod
    def expected_type(cls) -> Dict[str, List[str]]:
        r"""Method to add expected type for current processor input which
        would be checked before running the processor if
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {}

    def check_record_and_writes_record(self, input_pack: PackType):
        # pylint: disable=protected-access
        r"""Method to check type consistency if
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` is enabled
        for the pipeline and write record of the output type of the current
         processor to the datapack. If the expected type or its attributes
         doesn't exist in the datapack record of the previous pipeline
         component, an ``ExpectedRecordNotFound`` will be raised.

        Args:
            input_pack: The input datapack.
        """
        if self._check_type_consistency:
            expected_types = self.expected_type()
            # check if expected types are in input pack.
            for expected_t in expected_types:
                if expected_t not in input_pack._meta.record.keys():
                    raise ExpectedRecordNotFound(
                        f"The record type {expected_t} is not found in "
                        f"meta of the provided pack.")
                else:
                    expected_t_value = self.expected_type().get(expected_t)
                    if expected_t_value is not None:
                        for expected_t_v in expected_t_value:
                            if expected_t_v not in input_pack._meta.\
                                    record.get(expected_t, []):
                                raise ExpectedRecordNotFound(
                                    f"The record attribute type {expected_t_v} "
                                    f"is not found in attribute of record "
                                    f"{expected_t} in meta of the input "
                                    f"datapack.")
        # add type record of output to meta of the input pack.
        self.record(input_pack._meta.record)

    def process(self, input_pack: PackType):
        self.check_record_and_writes_record(input_pack)
        # Set the component for recording purpose.
        self._process(input_pack)

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

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        r"""Returns a `dict` of configurations of the processor with default
        values. Used to replace the missing values of input ``configs`` during
        pipeline construction.
        """
        config = super().default_configs()
        config.update({
            'selector': {
                'type': 'forte.data.selector.DummySelector',
                'args': None,
                'kwargs': {}
            },
            'overwrite': False,
        })
        return config
