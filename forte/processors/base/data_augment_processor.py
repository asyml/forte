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
Processors that augment the data.
"""
from abc import abstractmethod
from forte.common.resources import Resources
from forte.common.configuration import Config
from forte.processors.base.base_processor import BaseProcessor
__all__ = [
    "BaseDataAugmentProcessor",
    "ReplacementDataAugmentProcessor"
]


class BaseDataAugmentProcessor(BaseProcessor):
    r"""The base class of processors that augment data.
    This processor instantiates an augmenter where specific
    data augmentation algorithms are implemented. The augmenter
    will run the algorithms and the processor will create Forte
    data structures based on the augmented inputs.
    """
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

    @property
    def augmenter(self):
        return self._augmenter

    @augmenter.setter
    def augmenter(self, augmenter):
        r"""
        This function takes in the instantiated augmenter
        and bounds it to the processor.
        """
        # self.augmenter = augmenter
        self._augmenter = augmenter


    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - augment_entries: defines the entries that will be returned
            - aug_num: The number of augmented data for data augmentation. For example,
            if aug_num = 5, the processor will output a multipack with 1 original
            input + 5 augmented inputs.
            - kwargs: augmenter-specific parameters
        """
        config = super().default_configs()
        config.update({
            'augment_entries': ["Sentence", "Document"],
            'aug_num': 5,
            'type': "",
            'kwargs': {}
        })
        return config


class ReplacementDataAugmentProcessor(BaseDataAugmentProcessor):
    r"""
    Most of the Data Augmentation(DA) methods can be
    considered as replacement-based methods with different
    levels: character, word, sentence.
    """

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - replacement_level: defines the type of replacement(char/word/sentence),
            must be allowed by the the augmenter's algorithm. Specifically, the augmenter
            also has a list of allowed replacement_levels, and it must include this
            processor's replacement_level.
            - replacement_prob: defines the probability of replacing the original input.
        """
        config = super().default_configs()
        config.update({
            'replacement_level': 'word',
            'replacement_prob': 0.1,
        })
        return config