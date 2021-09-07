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

#import json
#from abc import abstractmethod

import random
import requests
from typing import Tuple, Union, Dict, Any

from forte.data.ontology import Annotation
from forte.processors.data_augment.algorithms.text_replacement_op import (
    TextReplacementOp,
)
from forte.common.configuration import Config

__all__ = [
    "UniformTypoGenerator",
    "TypoReplacementOp",
]


class UniformTypoGenerator():
    r"""
    A uniform generateor that generates a typo from a typo dictionary.

    Args:
        word: input word that needs to be replaced,
        dict_path: the url to the pre-defined typo json file. The key is a
            word we want to replace. The value is a list containing various
            typos of the corresponding key.
    """

    def generate(self, word: str, dict_path: str) -> str:
        # with open(dict_path, encoding="utf8") as json_file:
        #     data = json.load(json_file)
        r = requests.get(dict_path)
        data = r.json()
        if word in data.keys():
            result: str = random.choice(data[word])
            return result
        else:
            return word


class TypoReplacementOp(TextReplacementOp):
    r"""
    This class is a replacement op using a pre-defined
    spelling mistake dictionary to simulate spelling mistake.

    Args:
        configs:
            The config should contain
                `prob`(float): The probability of replacement, 
                    should fall in [0, 1].
                dict_path (str): the url to the pre-defined typo json file. 
                    The key is a word we want to replace. The value is a list 
                    containing various typos of the corresponding key.
                typo_generator (str): A generator that takes in a word and
                    outputs the replacement typo.
    """

    def __init__(
        self, configs: Union[Config, Dict[str, Any]]
    ):
        super().__init__(configs)
        self.typoGenerator = configs['typo_generator']
        if 'dict_path' in configs.keys():
            self.dict_path = configs['dict_path']
        else:
            # default typo dictionary
            self.dict_path = 'https://raw.githubusercontent.com/wanglec/' + \
                'temporaryJson/main/misspelling.json'

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
        if self.typoGenerator == 'UniformTypoGenerator':
            word: str = UniformTypoGenerator().generate(input_anno.text,
                                                        self.dict_path)
            return True, word
        else:
            raise ValueError(
                "The valid options for typo_generator are "
                "[UniformTypoGenerator]"
            )
