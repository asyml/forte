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

from forte.data.ontology import Annotation
from forte.processors.data_augment.algorithms.text_replacement_op import (
    TextReplacementOp,
)
from forte.common.configuration import Config
from forte.processors.data_augment.algorithms.typo_generator import TypoGenerator

__all__ = [
    "TypoReplacementOp",
]


class TypoReplacementOp(TextReplacementOp):
    r"""
    This class is a replacement op using a pre-defined
    spelling mistake dictionary to simulate spelling mistake.

    The configuration should have the following fields:

    Args:
        typoGenerator: A generator that outputs the replacement word.
        configs:
            The config should contain
                `prob`(float): The probability of replacement, should fall in [0, 1].
                dict_path (str): The absolute path to the typo json file for the
                    pre-defined spelling mistake.
    """

    def __init__(
        self, typoGenerator: TypoGenerator, configs: Union[Config, Dict[str, Any]]
    ):
        super().__init__(configs)
        self.typoGenerator = typoGenerator
        self.dict_path = configs['dict_path']

    def replace(self, input_anno: Annotation) -> Tuple[bool, str]:
        r"""
        This function replaces a word from a typo dictionary.

        Args:
            input_anno (Annotation): The input annotation.
        Returns:
            A tuple, where the first element is a boolean value indicating
            whether the replacement happens, and the second element is the
            replaced string.
        """
        # If the replacement does not happen, return False.
        if random.random() > self.configs.prob:
            return False, input_anno.text
        word: str = self.typoGenerator.generate(input_anno.text, self.dict_path)
        return True, word
