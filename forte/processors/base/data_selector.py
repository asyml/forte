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
"""
Processors that select datapack for data augmentation.
It is used after data reader but before data augment processor.
It selects a subset of the original data and then augment the selected data.
"""
import random
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import (
    Token, Sentence, Document
)
from forte.common.resources import Resources
from forte.common.configuration import Config


__all__ = [
    "BaseDataSelectorProcessor",
    "RandomSelectorProcessor",
    "LengthSelectorProcessor",
]


class BaseDataSelectorProcessor(PackProcessor):
    r"""
    The data selector processor for data augmentation.
    This class drops the datapack by the selection criteria.
    It takes a datapack as input and process to keep or drop it.
    """
    # def __init__(self, configs: Dict[str, str]):
    #     super().__init__(configs)
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

    def select(self, input_pack: DataPack) -> bool:
        raise NotImplementedError

    def _process(self, input_pack: DataPack):
        print(self.configs["max_length"])
        if not self.select(input_pack):
            # input_pack.__del__()    #Todo: remove datapack
            input_pack.set_text("")


class RandomSelectorProcessor(BaseDataSelectorProcessor):
    r"""
    The random data selector processor for data augmentation.
    This class simply drops the datapack according to a specific drop rate.
    It does not take the datapack textual information into consideration.
    """
    # def __init__(self, configs: Dict[str, str]):
    #     super().__init__(configs)
    #     random.seed(0)
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)


    def select(self, input_pack: DataPack) -> bool:
        prob:float = random.random()
        return prob > float(self.configs["drop_rate"])

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update({
            "drop_rate": 0.4,
        })
        return config


class LengthSelectorProcessor(BaseDataSelectorProcessor):
    r"""
    This class simply drops the datapack according to its document length.
    """
    # def __init__(self, configs: Dict[str, str]):
    #     super().__init__(configs)
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)


    def select(self, input_pack: DataPack) -> bool:
        text:str = input_pack.text
        return len(text) < int(self.configs["max_length"])

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update({
            "max_length": 100,
        })
        return config