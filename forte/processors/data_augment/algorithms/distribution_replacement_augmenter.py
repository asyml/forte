# Copyright 2020 The Forte Authors. All Rights Reserved.
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

from typing import Tuple
from forte.common.configuration import Config
from forte.data.ontology.core import Entry
from forte.processors.data_augment.algorithms.text_replacement_op \
    import TextReplacementOp
from forte.processors.data_augment.algorithms.sampler import Sampler


__all__ = [
    "DistributionReplacementOp",
]


class DistributionReplacementOp(TextReplacementOp):
    r"""
    This class is a replacement op sampling from a distribution
    to generate word-level new text.
    todo: subject to change if TextReplacementOp and ReplacementDataAugmentProcessor change.
    """
    def __init__(self, sampler: Sampler, configs: Config):
        super().__init__(configs)
        self.sampler = sampler

    # pylint: disable=unused-argument
    def replace(self, input: Entry) -> Tuple[bool, str]:
        r"""
        This function replaces a word from sampling a distribution.
        Args:
        Returns:
            a word sampled from distribution
        """
        word: str = self.sampler.sample()
        return True, word
