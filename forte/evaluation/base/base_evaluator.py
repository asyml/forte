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
Defines the Evaluator interface and related functions.
"""
from abc import abstractmethod
from typing import Any, Dict, Set

from forte.data.base_pack import PackType
from forte.pipeline_component import PipelineComponent
from forte.utils.utils_processor import record_types_and_attributes_check

__all__ = [
    "Evaluator",
]


class Evaluator(PipelineComponent[PackType]):
    r"""The base class of the evaluator."""
    def __init__(self):
        super().__init__()
        self._pred_pack_expectation: Dict[str, Set[str]] = None
        self._ref_pack_expectation: Dict[str, Set[str]] = None

    @abstractmethod
    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        r"""The actual consume function that will be called by the pipeline.
        This function will deal with the basic pipeline status and call the
        `consume_next` function.

        Args:
            pred_pack: The prediction pack, which should contain the system
                predicted results.
            ref_pack: The reference pack, which should contain the reference
                to score on.
        """

        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        r"""The evaluator gather the results and the score should be obtained
        here.
        """
        raise NotImplementedError

    def expected_types_and_attributes(self,
                                      pred_pack_expectation: Dict[
                                          str, Set[str]],
                                      ref_pack_expectation: Dict[
                                          str, Set[str]]):
        r"""If the evaluator has required input types and attributes for
        `pred_pack` or `ref_pack`, user could add the types and attributes
        required with this function.

        Args:
            pred_pack_expectation: The expected types and attributes of
                prediction pack.
            ref_pack_expectation: The expected types and attributes of
                reference pack.
        """
        self._pred_pack_expectation = pred_pack_expectation
        self._ref_pack_expectation = ref_pack_expectation

    def pred_pack_record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of prediction datapack of
        current processor to :attr:`forte.data.base_pack.BaseMeta.record`.

        Args:
            record_meta: The field in the datapack for type record that need to
                fill in for consistency checking.
        """
        pass

    def ref_pack_record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of reference datapack of
        current processor to :attr:`forte.data.base_pack.BaseMeta.record`.

        Args:
            record_meta: The field in the datapack for record that need to
                fill in for consistency checking.
        """
        pass

    def check_record(self, pred_pack: PackType, ref_pack: PackType):
        # pylint: disable=protected-access
        r"""Method to check type consistency if
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` is enabled
        for the pipeline. If any expected type or its attribute
        does not exist in the `pred_pack` or `ref_pack` record of the previous
        pipeline component, an error of
        :class:`~forte.common.exception.ExpectedRecordNotFound` will be raised.

        Args:
            pred_pack: The prediction pack, which should contain the system
                predicted results.
            ref_pack: The reference pack, which should contain the reference
                to score on.
        """
        if self._check_type_consistency:
            record_types_and_attributes_check(self._pred_pack_expectation,
                                              pred_pack)
            record_types_and_attributes_check(self._ref_pack_expectation,
                                              ref_pack)

    def writes_record(self, pred_pack: PackType, ref_pack: PackType):
        r"""Method to write records of the output type of the current
        processor to the datapack. The key of the record should be the entry
        type and values should be attributes of the entry type. All the
        information would be used for consistency checking purpose if
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` is enabled
        for the pipeline.

        Args:
            pred_pack: The prediction pack, which should contain the system
                predicted results.
            ref_pack: The reference pack, which should contain the reference
                to score on.

        """
        # pylint: disable=protected-access
        self.pred_pack_record(pred_pack._meta.record)
        self.ref_pack_record(ref_pack._meta.record)
