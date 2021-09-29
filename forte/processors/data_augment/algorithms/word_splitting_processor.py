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
Randomly choose n words (With length greater than 1) and split it at a random
position. Do this n times, where n = alpha * input length.
Example: Original Text -> "I will be there soon." ,
Augmented Text -> "I w ill be there so on."
"""

from math import ceil
import random
from typing import List, Iterable

from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology import Annotation
from forte.processors.data_augment import ReplacementDataAugmentProcessor
from forte.utils.utils import get_class

__all__ = ["RandomWordSplitDataAugmentProcessor"]


class RandomWordSplitDataAugmentProcessor(ReplacementDataAugmentProcessor):
    r"""
    This class creates a processor to perform Random Word Splitting.
    It randomly chooses n words in a sentence and splits each word at
    a random position where n = alpha * input length.
    """

    def _augment(self, input_pack: MultiPack, aug_pack_names: List[str]):
        augment_entry = get_class(self.configs["augment_entry"])

        for pack_name in aug_pack_names:
            data_pack: DataPack = input_pack.get_pack(pack_name)

            annotations: List[Annotation] = []
            indexes: List[int] = []
            endings = []
            annos: Iterable[Annotation] = data_pack.get(augment_entry)
            for idx, anno in enumerate(annos):
                annotations.append(anno)
                indexes.append(idx)
                endings.append(anno.end)
            if len(annotations) > 0:
                annotation_to_split = random.sample(
                    [
                        (anno, idx)
                        for (anno, idx) in zip(annotations, indexes)
                        if (anno.end - anno.begin) > 1
                    ],
                    ceil(self.configs["alpha"] * len(annotations)),
                )
                annotation_to_split = sorted(
                    annotation_to_split, key=lambda x: x[1], reverse=True
                )
                for curr_anno in annotation_to_split:
                    src_anno, src_idx = curr_anno
                    splitting_position = random.randrange(
                        1, (src_anno.end - src_anno.begin)
                    )
                    word_split = [
                        src_anno.text[:splitting_position],
                        src_anno.text[splitting_position:],
                    ]
                    if src_idx != 0:
                        first_position = endings[src_idx - 1] + 1
                        second_position = endings[src_idx]
                        word_split[1] = " " + word_split[1]
                    else:
                        first_position = 0
                        second_position = endings[0]
                        word_split[1] = " " + word_split[1]

                    self._insert(word_split[1], data_pack, second_position)
                    self._delete(src_anno)
                    self._insert(word_split[0], data_pack, first_position)

    @classmethod
    def default_configs(cls):
        """
        Returns:
        A dictionary with the default config for this processor.
        Additional keys for determining how many words will be split:
        - alpha: 0 <= alpha <= 1. indicates the percent of the words
        in a sentence that are changed. The processor will perform
        the Word Splitting operation 2 * (input length * alpha) times
        after deleting the original annotation.
        """
        config = super().default_configs()
        config.update(
            {
                "augment_entry": "ft.onto.base_ontology.Token",
                "other_entry_policy": {
                    "type": "",
                    "kwargs": {
                        "ft.onto.base_ontology.EntityMention": "auto_align",
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
