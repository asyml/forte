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
from typing import Any, Optional

from texar.torch import HParams

from forte.data.base_pack import PackType
from forte.pipeline_component import PipelineComponent

__all__ = [
    "Evaluator",
]


class Evaluator(PipelineComponent[PackType]):
    r"""The evaluator.

    Args:
        config: The configuration of the evaluator.
    """
    def __init__(self, config: Optional[HParams] = None):
        self.config: Optional[HParams] = config

    @abstractmethod
    def consume_next(self, pred_pack: PackType, ref_pack: PackType):
        r"""Consume the prediction pack and the reference pack to compute
        evaluation results.

        Args:
            pred_pack: The prediction datapack, which should contain the system
                predicted results.
            ref_pack: The reference datapack, which should contain the reference
                to score on.
        """

        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> Any:
        r"""The evaluator gather the results and the score can be obtained here.
        """
        raise NotImplementedError
