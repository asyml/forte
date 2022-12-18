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
import json
from typing import Tuple, Union, Dict, Any


from forte.data.ontology import Annotation
from forte.processors.data_augment.algorithms.single_annotation_op import (
    SingleAnnotationAugmentOp,
)
from forte.common.configuration import Config
from forte.utils import create_import_error_msg

__all__ = [
    "UniformTypoGenerator",
    "TypoReplacementOp",
]


class UniformTypoGenerator:
    r"""
    A uniform generator that generates a typo from a typo dictionary.

    Args:
        word: input word that needs to be replaced,
        dict_path: the url or the path to the pre-defined typo json file.
            The key is a word we want to replace. The value is a list
            containing various typos of the corresponding key.

            .. code-block:: python

                {
                    "apparent": ["aparent", "apparant"],
                    "bankruptcy": ["bankrupcy", "banruptcy"],
                    "barbecue": ["barbeque"]
                }
    """

    def __init__(self, dict_path: str):
        try:
            import requests  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                create_import_error_msg(
                    "requests", "data_aug", "data augment support"
                )
            ) from e

        try:
            r = requests.get(dict_path, timeout=30)
            self.data = r.json()
        except requests.exceptions.RequestException:
            with open(dict_path, encoding="utf8") as json_file:
                self.data = json.load(json_file)

    def generate(self, word: str) -> str:
        if word in self.data.keys():
            result: str = random.choice(self.data[word])
            return result
        else:
            return word


class TypoReplacementOp(SingleAnnotationAugmentOp):
    r"""
    This class is a replacement op using a pre-defined
    spelling mistake dictionary to simulate spelling mistake.
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]):
        super().__init__(configs)
        if "dict_path" in configs.keys():
            self.dict_path = configs["dict_path"]
        else:
            # default typo dictionary
            self.dict_path = (
                "https://raw.githubusercontent.com/wanglec/"
                + "temporaryJson/main/misspelling.json"
            )
        if configs["typo_generator"] == "uniform":
            self.typo_generator = UniformTypoGenerator(self.dict_path)
        else:
            raise ValueError(
                "The valid options for typo_generator are [uniform]"
            )

    def single_annotation_augment(
        self, input_anno: Annotation
    ) -> Tuple[bool, str]:
        r"""
        This function replaces a word from a typo dictionary.

        Args:
            input_anno: The input annotation.
        Returns:
            A tuple, where the first element is a boolean value indicating
            whether the replacement happens, and the second element is the
            replaced string.
        """
        # If the replacement does not happen, return False.
        if random.random() > self.configs.prob:
            return False, input_anno.text
        word: str = self.typo_generator.generate(input_anno.text)
        return True, word

    @classmethod
    def default_configs(cls):
        r"""
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - prob (float): The probability of replacement,
              should fall in [0, 1]. Default value is 0.1
            - dict_path (str): the `url` or the path to the pre-defined
              typo json file. The key is a word we want to replace.
              The value is a list containing various typos
              of the corresponding key.
            - typo_generator (str): A generator that takes in a word and
              outputs the replacement typo.
        """
        return {
            "prob": 0.1,
            "dict_path": "https://raw.githubusercontent.com/wanglec/"
            + "temporaryJson/main/misspelling.json",
            "typo_generator": "uniform",
        }
