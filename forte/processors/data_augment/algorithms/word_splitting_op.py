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
Data augmentation operation for the Random Word Splitting operation.
Randomly choose n words (With length greater than 1) and split it at a random
position. Do this n times, where n = alpha * input length.
Example: Original Text -> "I will be there soon." ,
Augmented Text -> "I w ill be there so on."
"""

from math import ceil
import random
from typing import List, Iterable

from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.processors.data_augment.algorithms.base_data_augmentation_op import (
    BaseDataAugmentationOp,
)
from forte.utils.utils import get_class

__all__ = ["RandomWordSplitDataAugmentOp"]


class RandomWordSplitDataAugmentOp(BaseDataAugmentationOp):
    r"""
    This class creates an operation to perform Random Word Splitting.
    It randomly chooses n words in a sentence and splits each word at
    a random position where n = alpha * input length.
    alpha indicates the percent of the words in a sentence that are changed.
    """

    def augment(self, data_pack: DataPack) -> bool:
        r"""
        This function splits a given word at a random position and replaces
        the original word with 2 split parts of it.
        """
        augment_entry = get_class(self.configs["augment_entry"])

        annotations: List[Annotation] = []
        indexes: List[int] = []
        endings = []
        annos: Iterable[Annotation] = data_pack.get(augment_entry)
        try:
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

                    self.insert_annotated_span(
                        word_split[1],
                        data_pack,
                        second_position,
                        self.configs["augment_entry"],
                    )

                    self.delete_annotation(src_anno)

                    self.insert_annotated_span(
                        word_split[0],
                        data_pack,
                        first_position,
                        self.configs["augment_entry"],
                    )
            return True
        except ValueError:
            return False

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.

        Additional keys for determining how many words will be split:
            - alpha (float):
                0 <= alpha <= 1. indicates the percent of the words
                in a sentence that are changed.
            - augment_entry (str):
                Defines the entry the processor will augment.
                It should be a full qualified name of the entry class.
                For example, "ft.onto.base_ontology.Sentence".
        """
        return {
            "augment_entry": "ft.onto.base_ontology.Token",
            "alpha": 0.1,
        }
