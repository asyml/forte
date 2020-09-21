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
from forte.processors.data_augment.algorithms.dictionary_replacement_augmenter import DictionaryReplacementAugmenter
from forte.processors.base.pack_processor import MultiPackProcessor
__all__ = [
    "BaseDataAugmentProcessor"
]


class BaseDataAugmentProcessor(MultiPackProcessor):
    r"""The base class of processors that augment data.
    This processor instantiates an augmenter where specific
    data augmentation algorithms are embedded. The augmenter
    will run the algorithms and the processor will pack the
    strings.

    The DA methods can all be considered as replacement-based
    methods with different levels: character, word, sentence.
    """
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.augmenter = self.get_augmenter()

    def get_augmenter(self):
        r"""
        This function parse the augment algorithm and
        instantiate an augmenter.
        :return: an instance of data augmenter.
        """
        algorithm = self.configs.augment_algorithm
        if algorithm == "DictionaryReplacement":
            lang = self.configs.get("lang", "eng")
            augmenter = DictionaryReplacementAugmenter(configs={"lang": lang})
        else:
            raise ModuleNotFoundError("The augment algorithm {} is not implemented!".format(algorithm))
        return augmenter


    @classmethod
    def default_configs(cls):
        """
        :return: A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - augment_algorithm: defines the augmenter to use
            - augment_ontologies: defines the ontologies that will be returned
            - replacement_prob: defines the probability of replacement
            - replacement_prob: defines the type of replacement(char/word/sentence),
            must align with(is included in) the replacement levels of the augmenter.
            - kwargs: augmenter-specific parameters
        """
        config = super().default_configs()
        config.update({
            'augment_algorithm': "DictionaryReplacement",
            'augment_ontologies': ["Sentence", "Document"],
            'replacement_prob': 0.1,
            'replacement_level': 'word',
            'type': "",
            'kwargs': {}
        })
        return config