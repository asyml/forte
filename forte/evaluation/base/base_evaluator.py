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
from typing import Any

from forte.data.base_pack import PackType
from forte.pipeline_component import PipelineComponent

__all__ = [
    "Evaluator",
]


class Evaluator(PipelineComponent[PackType]):
    r"""The base class of the evaluator."""

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
