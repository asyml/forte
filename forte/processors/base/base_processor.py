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
import logging
from abc import abstractmethod, ABC
from typing import Any, Dict, Set

from forte.data.base_pack import PackType
from forte.data.selector import DummySelector
from forte.pipeline_component import PipelineComponent
from forte.utils.utils_processor import (
    record_types_and_attributes_check,
    collect_input_pack_record,
)

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

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output record of the current processor to
        :attr:`forte.data.data_pack.Meta.record`. The key of the record
        should be the entry type and values should be attributes of the entry
        type. All the information would be used for consistency checking
        purpose if the pipeline is initialized with
        `enforce_consistency=True`.

        Args:
            record_meta: The field in the datapack for type record that need to
                fill in for consistency checking.
        """
        pass

    def expected_types_and_attributes(self) -> Dict[str, Set[str]]:
        r"""Method to add expected types and attributes for the input of the
        current processor which would be checked before running the processor if
        if the pipeline is initialized with
        `enforce_consistency=True`.
        """

        return {}

    def check_record(self, input_pack: PackType):
        # pylint: disable=protected-access
        r"""Method to check type consistency if the pipeline is initialized with
        `enforce_consistency=True`. If any expected type or its attribute
        does not exist in the datapack record of the previous pipeline
        component, an error of
        :class:`~forte.common.exception.ExpectedRecordNotFound` will be raised.

        Args:
            input_pack: The input datapack.
        """
        if self._check_type_consistency:
            expectation = self.expected_types_and_attributes()
            input_pack_record = collect_input_pack_record(
                self.resources, input_pack
            )
            record_types_and_attributes_check(expectation, input_pack_record)

    def write_record(self, input_pack: PackType):
        r"""Method to write records of the output type of the current
        processor to the datapack. The key of the record should be the entry
        type and values should be attributes of the entry type. All the
        information would be used for consistency checking purpose if
        the pipeline is initialized with
        `enforce_consistency=True`.

        Args:
            input_pack: The input datapack.

        """
        # pylint: disable=protected-access
        if self._check_type_consistency:
            try:
                self.record(input_pack._meta.record)
            except AttributeError:
                # For backward compatibility, no record to write.
                logging.info(
                    "Packs of the old format do not have the record field."
                )

    def process(self, input_pack: PackType):
        self.check_record(input_pack)
        # Set the component for recording purpose.
        self._process(input_pack)
        self.write_record(input_pack)

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
        config.update(
            {
                "selector": {
                    "type": "forte.data.selector.DummySelector",
                    "args": None,
                    "kwargs": {},
                },
                "overwrite": False,
            }
        )
        return config
