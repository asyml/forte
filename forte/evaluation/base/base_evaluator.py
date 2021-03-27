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
from typing import Any, Dict

from forte.data.base_pack import PackType
from forte.pipeline_component import PipelineComponent
from forte.common import ExpectedRecordNotFound

__all__ = [
    "Evaluator",
]


class Evaluator(PipelineComponent[PackType]):
    r"""The base class of the evaluator."""
    def __init__(self):
        super().__init__()
        self._pred_pack_expected_type: Dict = None
        self._ref_pack_expected_type: Dict = None

    @abstractmethod
    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        r"""The actual consume function that will be called by the pipeline.
        This function will deal with the basic pipeline status and call the
        `consume_next` function.

        Args:
            pred_pack: The prediction datapack, which should contain the system
                predicted results.
            ref_pack: The reference datapack, which should contain the reference
                to score on.
        """

        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        r"""The evaluator gather the results and the score should be obtained
        here.
        """
        raise NotImplementedError

    def expected_type(self, pred_pack_expected_type: Dict,
                      ref_pack_expected_type: Dict):
        r"""If the evaluator has required input type for pred_pack or ref_pack,
        user could add the type required with this function.

        """
        self._pred_pack_expected_type = pred_pack_expected_type
        self._ref_pack_expected_type = ref_pack_expected_type

    def pred_pack_record(self, record_meta: Dict):
        r"""Method to add output type record of prediction datapack of
        current processor to :attr:`forte.data.data_pack.Meta.record`
        """
        pass

    def ref_pack_record(self, record_meta: Dict):
        r"""Method to add output type record of reference datapack of
        current processor to :attr:`forte.data.data_pack.Meta.record`
        """
        pass

    def check_record_and_writes_record(self, pred_pack: PackType,
                                       ref_pack: PackType):
        # pylint: disable=protected-access
        r"""Method to check type consistency if
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` is enabled
        for the pipeline and write record of the output type of the current
         processor to the datapack. If the expected type or its attributes
         doesn't exist in the pred_pack or ref_pack record of the previous
         pipeline component, an ``ExpectedRecordNotFound`` will be raised.

        Args:
            pred_pack: The prediction datapack, which should contain the system
                predicted results.
            ref_pack: The reference datapack, which should contain the reference
                to score on.
        """
        if self._check_type_consistency:
            if self._pred_pack_expected_type is not None:
                # check if expected types are in input pack.
                for expected_t in self._pred_pack_expected_type:
                    if expected_t not in pred_pack._meta.record.keys():
                        raise ExpectedRecordNotFound(
                            f"The record type {expected_t} is not found in "
                            f"meta of the prediction datapack.")
                    else:
                        expected_value = self._pred_pack_expected_type.get(
                            expected_t)
                        for expected_t_v in expected_value:
                            if expected_t_v not in pred_pack._meta.record.get(
                                    expected_t):
                                raise ExpectedRecordNotFound(
                                    f"The record attribute type {expected_t_v}"
                                    f" is not found in attribute of record "
                                    f"{expected_t} in meta of the prediction "
                                    f"datapack.")
            if self._ref_pack_expected_type is not None:
                # check if expected types are in input pack.
                for expected_t in self._ref_pack_expected_type:
                    if expected_t not in ref_pack._meta.record.keys():
                        raise ExpectedRecordNotFound(
                            f"The record type {expected_t} is not found in "
                            f"meta of the reference datapack.")
                    else:
                        expected_value = self._ref_pack_expected_type.get(
                            expected_t)
                        for expected_t_v in expected_value:
                            if expected_t_v not in ref_pack._meta.record.get(
                                    expected_t):
                                raise ExpectedRecordNotFound(
                                    f"The record attribute type {expected_t_v}"
                                    f" is not found in attribute of record "
                                    f"{expected_t} in meta of the reference "
                                    f"datapack.")
        # add type record of output to meta of the input pack.
        self.pred_pack_record(pred_pack._meta.record)
        self.ref_pack_record(ref_pack._meta.record)
