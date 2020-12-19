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

from math import ceil
import random
from typing import List, Dict

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base import ReplacementDataAugmentProcessor
from forte.utils.utils import get_class, create_class_with_kwargs

__all__ = [
    "RandomSwapDataAugmentProcessor",
    "RandomInsertionDataAugmentProcessor",
    "RandomDeletionDataAugmentProcessor",
]


english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
    'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
    'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"]


class RandomSwapDataAugmentProcessor(ReplacementDataAugmentProcessor):
    r"""
    Data augmentation processor for the Random Swap operation.
    Randomly choose two words in the sentence and swap their positions.
    Do this n times, where n = alpha * input length.
    """

    def _augment(self, input_pack: MultiPack, aug_pack_names: List[str]):
        augment_entry = get_class(self.configs["augment_entry"])
        for pack_name in aug_pack_names:
            data_pack: DataPack = input_pack.get_pack(pack_name)
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
                    self._replaced_annos[pid]\
                        .add((annotations[idx].span,
                                 annotations[replace_map[idx]].text))

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
            Additional keys for Random Swap:
            - alpha: indicates the percent of the words in a sentence
                are changed. The processor will perform the Random Swap
                operation (input length * alpha) times.
        """
        config = super().default_configs()
        config.update({
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "kwargs": {
                    "ft.onto.base_ontology.Document": "auto_align",
                    "ft.onto.base_ontology.Sentence": "auto_align"
                }
            },
            'alpha': 0.1,
            'augment_pack_names': {
                'type': '',
                'kwargs': {
                    'input_src': 'augmented_input_src'
                }
            }
        })
        return config


class RandomInsertionDataAugmentProcessor(ReplacementDataAugmentProcessor):
    r"""
    Data augmentation processor for the Random Insertion operation.
    Find a random synonym of a random word in the sentence that is
    not a stop word. Insert that synonym into a random position in
    the sentence. Do this n times, where n = alpha * input length.
    """
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.stopwords = set(configs['stopwords'])

    def _augment(self, input_pack: MultiPack, aug_pack_names: List[str]):
        replacement_op = create_class_with_kwargs(
            self.configs["data_aug_op"],
            class_args={
                "configs": self.configs["data_aug_op_config"]["kwargs"]
            }
        )
        augment_entry = get_class(self.configs["augment_entry"])

        for pack_name in aug_pack_names:
            data_pack: DataPack = input_pack.get_pack(pack_name)
            annotations = []
            pos = [0]
            for anno in data_pack.get(augment_entry):
                if anno.text not in self.stopwords:
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
                    self._insert(replaced_text, data_pack, insert_pos)

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
            By default, we use Dictionary Replacement with Wordnet to get
            synonyms to insert.
            Additional keys for Random Swap:
            - alpha: indicates the percent of the words in a sentence are
                changed. The processor will perform the Random Insertion
                operation (input length * alpha) times.
            - stopwords: a list of stopword for the language.
        """
        config = super().default_configs()
        config.update({
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                'kwargs': {
                    "ft.onto.base_ontology.Document": "auto_align",
                    "ft.onto.base_ontology.Sentence": "auto_align"
                }
            },
            'data_aug_op':
                "forte.processors.data_augment.algorithms."
                "dictionary_replacement_op.DictionaryReplacementOp",
            'data_aug_op_config': {
                "kwargs": {
                    "dictionary_class": (
                        "forte.processors.data_augment."
                        "algorithms.dictionary.WordnetDictionary"
                    ),
                    "prob": 1.0,
                    "lang": "eng",
                },
            },
            'alpha': 0.1,
            'augment_pack_names': {
                'type': '',
                'kwargs': {
                    'input_src': 'augmented_input_src'
                }
            },
            'stopwords': english_stopwords,
        })
        return config


class RandomDeletionDataAugmentProcessor(ReplacementDataAugmentProcessor):
    r"""
    Data augmentation processor for the Random Insertion operation.
    Randomly remove each word in the sentence with probability alpha.
    """

    def _augment(self, input_pack: MultiPack, aug_pack_names: List[str]):
        augment_entry = get_class(self.configs["augment_entry"])

        for pack_name in aug_pack_names:
            data_pack: DataPack = input_pack.get_pack(pack_name)
            for anno in data_pack.get(augment_entry):
                if random.random() < self.configs['alpha']:
                    self._delete(anno)

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
            Additional keys for Random Deletion:
            - alpha: the probability to delete each word
        """
        config = super().default_configs()
        config.update({
            'augment_entry': "ft.onto.base_ontology.Token",
            'other_entry_policy': {
                "kwargs": {
                    "ft.onto.base_ontology.Document": "auto_align",
                    "ft.onto.base_ontology.Sentence": "auto_align"
                }
            },
            "data_aug_op_config": {
                'kwargs': {}
            },
            "alpha": 0.1,
            'augment_pack_names': {
                'type': '',
                'kwargs': {
                    'input_src': 'augmented_input_src'
                }
            }
        })
        return config
