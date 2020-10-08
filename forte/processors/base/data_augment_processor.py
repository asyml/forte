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
Processors that augment the data. The processor will call
augmenters to generate texts similar to those in the input pack
and insert them to the original pack.
"""

from forte.processors.base.base_processor import BaseProcessor
from forte.processors.data_augment.algorithms.base_augmenter \
    import BaseDataAugmenter

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
    def __init__(self):
        super().__init__()
        self._augmenter = None

    def set_augmenter(self, augmenter: BaseDataAugmenter):
        r"""
        This function takes in the instantiated augmenter
        and bounds it to the processor.
        """
        self._augmenter = augmenter

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - aug_num: The number of augmented data for data augmentation.
            For example, if aug_num = 5, the processor will output
            a multipack with 1 original input + 5 augmented inputs.
            - kwargs: augmenter-specific parameters
        """
        config = super().default_configs()
        config.update({
            'aug_num': 5,
        })
        return config


class ReplacementDataAugmentProcessor(BaseDataAugmentProcessor):
    r"""
    Most of the Data Augmentation(DA) methods can be
    considered as replacement-based methods with different
    levels: character, word, sentence or document.
    """

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - augment_entries: defines the entries the processor
            will augment. It should be a full path to the entry class.
            - other_entry_policy: the policy for other entries that
            is not the "augment_entry".
            If "Delete", all the other entries will be erased.
            Otherwise, they will be kept as they were,
            but they might become invalid after the augmentation.
        """
        config = super().default_configs()
        config.update({
            'augment_entry': "ft.onto.base_ontology.Sentence",
            'other_entry_policy': "Delete"
        })
        return config
