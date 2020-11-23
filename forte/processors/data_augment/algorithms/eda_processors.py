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
Data augmentation processors from the paper "EDA: Easy Data Augmentation
Techniques for Boosting Performance on Text Classification Tasks", including
Random Swap, Random Insertion and Random Deletion. All three processors are
implemented based on the ReplacementDataAugmentProcessor.
"""

import random
from typing import List, Tuple, Dict
from nltk.corpus import stopwords
from math import ceil

from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base import ReplacementDataAugmentProcessor
from forte.utils.utils import get_class, create_class_with_kwargs

__all__ = [
    "RandomSwapDataAugmentProcessor",
    "RandomInsertionDataAugmentProcessor",
    "RandomDeletionDataAugmentProcessor",
]


class RandomSwapDataAugmentProcessor(ReplacementDataAugmentProcessor):
    r"""
    Data augmentation processor for the Random Swap operation.
    Randomly choose two words in the sentence and swap their positions.
    Do this n times.
    """

    def _process(self, input_pack: MultiPack):
        augment_entry = get_class(self.configs["augment_entry"])
        new_packs: List[Tuple[str, DataPack]] = []

        for pack_name, data_pack in input_pack.iter_packs():
            annotations = list(data_pack.get(augment_entry))
            if len(annotations) > 0:
                replace_map: Dict = {}
                for _ in range(ceil(self.configs['alpha'] * len(annotations))):
                    swap_idx = random.sample(range(len(annotations)), 2)
                    new_idx_0 = swap_idx[1] if swap_idx[1] not in replace_map \
                        else replace_map[swap_idx[1]]
                    new_idx_1 = swap_idx[0] if swap_idx[0] not in replace_map \
                        else replace_map[swap_idx[0]]
                    replace_map[swap_idx[0]] = new_idx_0
                    replace_map[swap_idx[1]] = new_idx_1
                pid: int = data_pack.pack_id
                for idx in replace_map:
                    self.replaced_annos[pid]\
                        .append((annotations[idx].span,
                                 annotations[replace_map[idx]].text))

            new_pack_name = "augmented_" + pack_name
            new_pack = self.auto_align_annotations(
                data_pack=data_pack,
                replaced_annotations=self.replaced_annos[
                    data_pack.meta.pack_id
                ]
            )
            new_packs.append((new_pack_name, new_pack))

        for new_pack_name, new_pack in new_packs:
            input_pack.add_pack_(new_pack, new_pack_name)

        self._copy_multi_pack_links(input_pack)

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
            Additional keys for Random Swap:
            - n: the number of times we perform the Random Swap operation
        """
        config = super().default_configs()
        config.update({
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "entry": [
                    "ft.onto.base_ontology.Document",
                    "ft.onto.base_ontology.Sentence"
                ],
                "policy": ["auto_align", "auto_align"],
            },
            "kwargs": {},
            'alpha': 0.1
        })
        return config


class RandomInsertionDataAugmentProcessor(ReplacementDataAugmentProcessor):
    r"""
    Data augmentation processor for the Random Insertion operation.
    Find a random synonym of a random word in the sentence that is
    not a stop word. Insert that synonym into a random position in
    the sentence. Do this n times.
    """

    def _process(self, input_pack: MultiPack):
        replacement_op = create_class_with_kwargs(
            self.configs["kwargs"]["data_aug_op"],
            class_args={
                "configs": self.configs["kwargs"]["data_aug_op_config"]
            }
        )
        augment_entry = get_class(self.configs["augment_entry"])
        new_packs: List[Tuple[str, DataPack]] = []

        for pack_name, data_pack in input_pack.iter_packs():
            annotations = []
            pos = [0]
            for anno in data_pack.get(augment_entry):
                if anno.text not in stopwords.words('english'):
                    annotations.append(anno)
                    pos.append(anno.end)
            if len(annotations) > 0:
                for _ in range(ceil(self.configs['alpha'] * len(annotations))):
                    src_anno = random.choice(annotations)
                    _, replaced_text = replacement_op.replace(src_anno)
                    insert_pos = random.choice(pos)
                    if insert_pos > 0:
                        replaced_text = " " + replaced_text
                    else:
                        replaced_text = replaced_text + " "
                    self.insert(replaced_text, data_pack, insert_pos)

            new_pack_name = "augmented_" + pack_name
            new_pack = self.auto_align_annotations(
                data_pack=data_pack,
                replaced_annotations=self.replaced_annos[
                    data_pack.meta.pack_id
                ]

            )
            new_packs.append((new_pack_name, new_pack))

        for new_pack_name, new_pack in new_packs:
            input_pack.add_pack_(new_pack, new_pack_name)

        self._copy_multi_pack_links(input_pack)

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
            By default, we use Dictionary Replacement with Wordnet to get
            synonyms to insert.
            Additional keys for Random Swap:
            - n: the number of times we perform the Random Insertion operation
        """
        config = super().default_configs()
        config.update({
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "entry": [
                    "ft.onto.base_ontology.Document",
                    "ft.onto.base_ontology.Sentence"
                ],
                "policy": ["auto_align", "auto_align"],
            },
            "kwargs": {
                'data_aug_op':
                    "forte.processors.data_augment.algorithms."
                    "dictionary_replacement_op.DictionaryReplacementOp",
                'data_aug_op_config': {
                    "dictionary": (
                        "forte.processors.data_augment."
                        "algorithms.dictionary.WordnetDictionary"
                    ),
                    "prob": 1.0,
                    "lang": "eng",
                }
            },
            'alpha': 0.1,
        })
        return config


class RandomDeletionDataAugmentProcessor(ReplacementDataAugmentProcessor):
    r"""
    Data augmentation processor for the Random Insertion operation.
    Randomly remove each word in the sentence with probability p.
    """

    def _process(self, input_pack: MultiPack):
        augment_entry = get_class(self.configs["augment_entry"])
        new_packs: List[Tuple[str, DataPack]] = []

        for pack_name, data_pack in input_pack.iter_packs():
            for anno in data_pack.get(augment_entry):
                if random.random() < self.configs['alpha']:
                    self.delete(anno)
            new_pack_name = "augmented_" + pack_name
            new_pack = self.auto_align_annotations(
                data_pack=data_pack,
                replaced_annotations=self.replaced_annos[
                    data_pack.meta.pack_id
                ]
            )
            new_packs.append((new_pack_name, new_pack))

        for new_pack_name, new_pack in new_packs:
            input_pack.add_pack_(new_pack, new_pack_name)

        self._copy_multi_pack_links(input_pack)

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
            Additional keys for Random Deletion:
            - prob: the probability to delete each word
        """
        config = super().default_configs()
        config.update({
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "entry": [
                    "ft.onto.base_ontology.Document",
                    "ft.onto.base_ontology.Sentence"
                ],
                "policy": ["auto_align", "auto_align"],
            },
            'alpha': 0.1,
        })
        return config
