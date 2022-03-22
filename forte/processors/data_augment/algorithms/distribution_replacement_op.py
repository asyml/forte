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
from forte.utils.utils import create_class_with_kwargs

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

        - `sampler_data`:
          A dictionary representing the configurations
          required to create the required sampler.

            - type:
              The type of sampler to be used (pass the
              path of the class which defines the required sampler)

            - kwargs:
              This dictionary contains the data that is to be
              fed to the required sampler. 2 possible values are
              `sampler_data` and `data_path`.If both parameters are passed,
              the data read from the file pointed to by `data_path` will be
              considered.

                - `sampler_data`:
                  Raw input to the sampler, This will be passed as the
                  `sampler_data`
                  config to the required sampler.

                - `data_path`:
                  The path to the file that contains the
                  the input that will be given to the sampler. For example,
                  when using `UniformSampler`, `data_path` will point to a file
                  (or `URl`) containing a list of values to be used as
                  `sampler_data` in `UniformSampler`.

            .. code-block:: python

                {
                    "type": "forte.processors.data_augment.algorithms.sampler.UniformSampler",
                    "kwargs":{
                        "sample": ["apple", "banana", "orange"]
                    }
                }
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]):
        super().__init__(configs)
        self.configs = self.make_configs(configs)
        self.cofigure_sampler()

    def replace(self, input_anno: Annotation) -> Tuple[bool, str]:
        r"""
        This function replaces a word by sampling from a distribution.

        Args:
            input_anno (Annotation): the input annotation.

        Returns:
            A tuple of two values, where the first element is a boolean value
            indicating whether the replacement happens, and the second
            element is the replaced word.
        """
        if random.random() > self.configs.prob:
            return False, input_anno.text
        word: str = self.sampler.sample()
        return True, word

    def cofigure_sampler(self) -> None:
        r"""
        This function sets the sampler that will be
        used by the distribution replacement op. The sampler will be set
        according to the configuration values
        """
        try:
            if "data_path" in self.configs["sampler_config"]["kwargs"].keys():
                distribution_path = self.configs["sampler_config"]["kwargs"][
                    "data_path"
                ]
                try:
                    r = requests.get(distribution_path)
                    data = r.json()
                except requests.exceptions.RequestException:
                    with open(distribution_path, encoding="utf8") as json_file:
                        data = json.load(json_file)
            else:
                data = self.configs["sampler_config"]["kwargs"]["sampler_data"]

            self.sampler = create_class_with_kwargs(
                self.configs["sampler_config"]["type"],
                {
                    "configs": {
                        "sampler_data": data,
                    }
                },
            )
        except KeyError as error:
            print("Could not configure Sampler: " + repr(error))

    @classmethod
    def default_configs(cls):
        return {
            "prob": 0.1,
            "sampler_config": {
                "type": "forte.processors.data_augment.algorithms.sampler.UniformSampler",
                "kwargs": {"sampler_data": []},
            },
        }
