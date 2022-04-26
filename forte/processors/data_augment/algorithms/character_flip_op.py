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
from typing import Tuple, Any, Dict, Union


from forte.data.ontology import Annotation
from forte.common.configuration import Config
from forte.processors.data_augment.algorithms.single_annotation_op import (
    SingleAnnotationAugmentOp,
)
from forte.utils import create_import_error_msg

__all__ = ["CharacterFlipOp"]


class CharacterFlipOp(SingleAnnotationAugmentOp):
    r"""
    A uniform generator that randomly flips a character with a similar
    looking character from a predefined dictionary imported from
    `"https://github.com/facebookresearch/AugLy/blob/main/" +
    "augly/text/augmenters/utils.py"`.
    (For example: the cat drank milk -> t/-/3 c@t d12@nk m!|_1<).

    Args:
        string: input string whose characters need to be replaced,

        dict_path: the `url` or the path to the pre-defined
                    typo json file,

        configs: prob(float): The probability of replacement,
                    should fall in [0, 1].
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]) -> None:
        try:
            import requests  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                create_import_error_msg(
                    "requests", "data_aug", "data augment support"
                )
            ) from e
        super().__init__(configs)

        self.dict_path = self.configs["dict_path"]

        try:
            r = requests.get(self.dict_path)
            self.data = r.json()
        except requests.exceptions.RequestException:
            with open(self.dict_path, encoding="utf8") as json_file:
                self.data = json.load(json_file)

    def _flip(self, char: str):
        r"""
        Flips character with similar character from input dictionary.

        Args:
            char: input character.
        Returns:
            the modified character.
        """
        if char in self.data:
            return random.choice(self.data[char])
        else:
            return char

    def single_annotation_augment(
        self, input_anno: Annotation
    ) -> Tuple[bool, str]:
        r"""
        Takes in the annotated string and performs the character
        flip operation on it that randomly augments few characters
        from it based on the probability value in the configs.

        Args:
            input_anno: the input annotation.
        Returns:
            A tuple with the first element being a boolean value indicating
            whether the replacement happens, and the second element is the
            final augmented string.
        """
        augmented_string = ""
        for char in input_anno.text:
            if char == " " or random.random() > self.configs["prob"]:
                augmented_string += char
            else:
                augmented_string += self._flip(char)
        return True, augmented_string

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - dict_path (str):
                the `url` or the path to the pre-defined
                typo `json` file. One example dictionary is provided
                at `https://raw.githubusercontent.com/ArnavParekhji`
                `/temporaryJson/main/character_flip.json`
                as a default value.
            - prob (float):
                The probability of replacement. This value
                should fall in [0, 1]. Default value is 0.1
        """
        return {
            "dict_path": (
                "https://raw.githubusercontent.com/ArnavParekhji/"
                + "temporaryJson/main/character_flip.json"
            ),
            "prob": 0.1,
        }
