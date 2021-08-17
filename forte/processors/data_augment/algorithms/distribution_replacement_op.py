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
import random
from typing import Tuple, Union, Dict, Any
from forte.common.configuration import Config
from forte.data.ontology import Annotation
from forte.processors.data_augment.algorithms.text_replacement_op import (
    TextReplacementOp,
)
from forte.processors.data_augment.algorithms.sampler import Sampler

__all__ = [
    "DistributionReplacementOp",
]


class DistributionReplacementOp(TextReplacementOp):
    r"""
    This class is a replacement op to replace the input word
    with a new word that is sampled by a sampler from a distribution.

    Args:
        sampler: The sampler that samples a word from a distribution.
        configs: The config should contain `prob`,
            The probability of whether to replace the input,
            it should fall in [0, 1].
    """

    def __init__(
        self, sampler: Sampler, configs: Union[Config, Dict[str, Any]]
    ):
        super().__init__(configs)
        self.sampler = sampler

    def replace(self, input_anno: Annotation) -> Tuple[bool, str]:
        r"""
        This function replaces a word by sampling from a distribution.

        Args:
            input_anno (Annotation): The input annotation.
        Returns:
            A tuple of two values, where the first element is a boolean value
            indicating whether the replacement happens, and the second
            element is the replaced word.
        """
        if random.random() > self.configs.prob:
            return False, input_anno.text
        word: str = self.sampler.sample()
        return True, word
