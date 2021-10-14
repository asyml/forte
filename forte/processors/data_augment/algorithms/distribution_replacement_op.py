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
import json
import random
from typing import Tuple, Union, Dict, Any
import requests
from forte.common.configurable import Configurable
from forte.common.configuration import Config
from forte.data.ontology import Annotation
from forte.processors.data_augment.algorithms.text_replacement_op import (
    TextReplacementOp,
)
from forte.processors.data_augment.algorithms.sampler import (
    UniformSampler,
    UnigramSampler,
)

__all__ = [
    "DistributionReplacementOp",
]


class DistributionReplacementOp(TextReplacementOp, Configurable):
    r"""
    This class is a replacement op to replace the input word
    with a new word that is sampled by a sampler from a distribution.

    Config Values:
        - `prob`:
            The probability of whether to replace the
            input, it should fall in `[0, 1]`.

        - `distribution_path`:
            A string representing the destination of data that will
            serve as input to the sampler. Default will be an empty string.
            The data must be stored in a json file as list for uniform sampler
            and dictionary for unigram sampler.

        - `sampler_type`:
            The type of sampler. It should be one
            of `("uniform", "unigram")`

        - `uniform_sampler_data`:
            If the data is to be passed directly, it is passed as a list
            to initialize a uniform sampler using this key.

        - `unigram_sampler_data`:
            If the data is to be passed directly, it is passed as a dict
            to initialize a unigram sampler using this key.
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]):
        super().__init__(configs)
        self.configs = self.make_configs(configs)
        if not self.cofigure_sampler():
            raise Exception("The sampler could not be created.")

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

    def cofigure_sampler(self) -> bool:
        r"""
        This function sets the sampler (Unigram or Uniform) that will be
        used by the distribution replacement op. The sampler will be set
        according to the configuration values

        Returns:
            A Boolean value indicating if the creation of the sampler was successful or not.

        """
        sampler_type = self.configs["sampler_type"]
        if sampler_type not in {"uniform", "unigram"}:
            raise ValueError(
                "The value of 'sampler_type' has to be one of ['uniform', 'unigram']."
            )

        distribution_path = self.configs["distribution_path"]
        if distribution_path:
            try:
                r = requests.get(distribution_path)
                data = r.json()
            except requests.exceptions.RequestException:
                with open(distribution_path, encoding="utf8") as json_file:
                    data = json.load(json_file)
        else:
            data = (
                self.configs["unigram_sampler_data"].__dict__["_hparams"]
                if sampler_type == "unigram"
                else self.configs["uniform_sampler_data"]
            )

        if sampler_type == "uniform":
            if not isinstance(data, list):
                raise TypeError("The input for uniform sampler must be a list")
            self.sampler = UniformSampler(
                configs={"uniform_sampler_word_list": data}
            )
            return True
        if sampler_type == "unigram":
            if not isinstance(data, dict):
                raise TypeError(
                    "The input for unigram sampler must be a dictionary"
                )
            self.sampler = UnigramSampler(
                configs={"unigram_dict": {"type": "", "kwargs": data}}
            )
            return True
        return False

    @classmethod
    def default_configs(cls):
        return {
            "prob": 0.1,
            "sampler_type": "uniform",
            "distribution_path": "",
            "uniform_sampler_data": [],
            "unigram_sampler_data": {},
            "@no_typecheck": "unigram_sampler_data",
        }
