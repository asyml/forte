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
Data augmentation processor for the Random Word Splitting operation.
Randomly choose n words (With length greater than 1) and split it at a random position.
Do this n times, where n = alpha * input length.
"""

from math import ceil
import random
from typing import List, Iterable

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Annotation
from forte.processors.data_augment import ReplacementDataAugmentProcessor
from forte.utils.utils import get_class

__all__ = ["RandomWordSplitDataAugmentProcessor"]


class RandomWordSplitDataAugmentProcessor(ReplacementDataAugmentProcessor):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

    def _augment(self, input_pack: MultiPack, aug_pack_names: List[str]):
        augment_entry = get_class(self.configs["augment_entry"])

        for pack_name in aug_pack_names:
            data_pack: DataPack = input_pack.get_pack(pack_name)
            annotations: List[Annotation] = []
            pos = []
            annos: Iterable[Annotation] = data_pack.get(augment_entry)
            for anno in annos:
                if len(anno.text) > 1:
                    annotations.append((anno, anno.end))
                    pos.append(anno.end)
            if len(annotations) > 0:
                for _ in range(ceil(self.configs["alpha"] * len(annotations))):
                    annotation_to_split = random.choice(annotations)
                    src_anno = annotation_to_split[0]
                    insert_pos = annotation_to_split[1]
                    splitting_position = random.randrange(1, len(src_anno.text))
                    word_split = [
                        src_anno.text[:splitting_position],
                        src_anno.text[splitting_position:],
                    ]
                    if insert_pos != pos[-1]:
                        word_split[1] = word_split[1] + " "

                    first_position = insert_pos
                    second_position = insert_pos + 1

                    self._delete(src_anno)
                    self._insert(word_split[0], data_pack, first_position)
                    self._insert(word_split[1], data_pack, second_position)

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
            Additional keys for determining how many words will be split:
            - alpha: 0 <= alpha <= 1. indicates the percent of the words
                in a sentence that are changed. The processor will perform
                the Word Splitting operation 2 * (input length * alpha) times after deleting the original annotation.
        """
        config = super().default_configs()
        config.update(
            {
                "augment_entry": "ft.onto.base_ontology.Token",
                "other_entry_policy": {
                    "type": "",
                    "kwargs": {
                        "ft.onto.base_ontology.Document": "auto_align",
                        "ft.onto.base_ontology.Sentence": "auto_align",
                    },
                },
                "data_aug_op": "forte.processors.data_augment.algorithms.",
                "alpha": 0.1,
                "augment_pack_names": {
                    "type": "",
                    "kwargs": {"input_src": "augmented_input_src"},
                },
            }
        )
        return config
