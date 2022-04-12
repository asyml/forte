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
from typing import List, Dict, Iterable, Union, Any
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.processors.data_augment.algorithms.base_data_augmentation_op import (
    BaseDataAugmentationOp,
)
from forte.utils.utils import get_class, create_class_with_kwargs

__all__ = [
    "RandomSwapDataAugmentOp",
    "RandomInsertionDataAugmentOp",
    "RandomDeletionDataAugmentOp",
]

english_stopwords = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
]


class RandomSwapDataAugmentOp(BaseDataAugmentationOp):
    r"""
    Data augmentation operation for the Random Swap operation.
    Randomly choose two words in the sentence and swap their positions.
    Do this n times, where n = alpha * input length.
    """

    def augment(self, data_pack: DataPack) -> bool:
        augment_entry = get_class(self.configs["augment_entry"])
        if not issubclass(augment_entry, Annotation):
            raise ValueError(
                f"This augmenter only accept data of "
                f"'forte.data.ontology.top.Annotation' type, "
                f"but {self.configs['augment_entry']} is not."
            )

        annotations: List[Annotation] = list(
            data_pack.get(self.configs["augment_entry"])
        )
        if len(annotations) > 0:
            replace_map: Dict = {}
        for _ in range(ceil(self.configs["alpha"] * len(annotations))):
            swap_idx = random.sample(range(len(annotations)), 2)
            new_idx_0 = (
                swap_idx[1]
                if swap_idx[1] not in replace_map
                else replace_map[swap_idx[1]]
            )
            new_idx_1 = (
                swap_idx[0]
                if swap_idx[0] not in replace_map
                else replace_map[swap_idx[0]]
            )
            replace_map[swap_idx[0]] = new_idx_0
            replace_map[swap_idx[1]] = new_idx_1

        for idx, replace_target in replace_map.items():
            try:
                self.replace_annotations(
                    annotations[idx], annotations[replace_target].text
                )
            except ValueError:
                return False
        return True

    @classmethod
    def default_configs(cls):
        """
        Additional keys for Random Swap:

            - augment_entry (str):
                Defines the entry the processor will augment.
                It should be a full qualified name of the entry class.
                For example, "ft.onto.base_ontology.Sentence".

            - alpha:
                0 <= alpha <= 1. indicates the percent of the words
                in a sentence that are changed. The processor will perform
                the Random Swap operation (input length * alpha) times.
                Default Value is 0.1.

        Returns:
            A dictionary with the default config for this processor.
        """
        return {
            "augment_entry": "ft.onto.base_ontology.Token",
            "other_entry_policy": {
                "ft.onto.base_ontology.Document": "auto_align",
                "ft.onto.base_ontology.Sentence": "auto_align",
            },
            "alpha": 0.1,
        }


class RandomInsertionDataAugmentOp(BaseDataAugmentationOp):
    r"""
    Data augmentation operation for the Random Insertion operation.
    Find a random synonym of a random word in the sentence that is
    not a stop word. Insert that synonym into a random position in
    the sentence. Do this n times, where n = alpha * input length.
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]) -> None:
        super().__init__(configs)
        self.stopwords = set(self.configs["stopwords"])

    def augment(self, data_pack: DataPack) -> bool:

        replacement_op = create_class_with_kwargs(
            self.configs["insertion_op_configs"]["type"],
            class_args={
                "configs": self.configs["insertion_op_configs"]["kwargs"]
            },
        )

        annotations: List[Annotation] = []
        pos = [0]
        annos: Iterable[Annotation] = data_pack.get(
            self.configs["augment_entry"]
        )
        for anno in annos:
            if anno.text not in self.stopwords:
                annotations.append(anno)
                pos.append(anno.end)
        if len(annotations) > 0:
            for _ in range(ceil(self.configs["alpha"] * len(annotations))):
                src_anno = random.choice(annotations)
                try:
                    _, replaced_text = replacement_op.single_annotation_augment(
                        src_anno
                    )
                except ValueError:
                    return False

                insert_pos = random.choice(pos)
                if insert_pos > 0:
                    replaced_text = " " + replaced_text
                else:
                    replaced_text = replaced_text + " "

                try:
                    self.insert_annotated_span(
                        replaced_text,
                        data_pack,
                        insert_pos,
                        self.configs["augment_entry"],
                    )
                except ValueError:
                    return False
        return True

    @classmethod
    def default_configs(cls):
        """
        Additional keys for Random Swap:

        - augment_entry (str):
            Defines the entry the processor will augment.
            It should be a full qualified name of the entry class.
            For example, "ft.onto.base_ontology.Sentence".

        - alpha:
            0 <= alpha <= 1. indicates the percent of the words
            in a sentence that are changed. The processor will perform
            the Random Insertion operation (input length * alpha) times.
            Default Value is 0.1

        - stopwords:
            a list of stopword for the language.

        - `insertion_op_config`:
            A dictionary representing the configurations
            required operation to take random annotations
            from the source data pack, augment them based
            on specified rules and insert them in random
            positions.

            - type:
                The type of data augmentation operation to be used
                (pass the path of the class which defines the
                required operation)

            - kwargs:
                This dictionary contains the data that is to be
                fed to the required operation (Make sure to be
                well versed with the required configurations of the
                operation that is defined in the type config).

            .. code-block:: python

                {
                    "type": "forte.processors.data_augment.algorithms."
                    "dictionary_replacement_op.DictionaryReplacementOp",
                    "kwargs":{
                        "dictionary_class": (
                            "forte.processors.data_augment."
                            "algorithms.dictionary.WordnetDictionary"
                        ),
                        "prob": 1.0,
                        "lang": "eng",
                    }
                }

        Returns:
            A dictionary with the default config for this processor.
            By default, we use Dictionary Replacement with Wordnet to get
            synonyms to insert.
        """
        return {
            "augment_entry": "ft.onto.base_ontology.Token",
            "other_entry_policy": {
                "ft.onto.base_ontology.Document": "auto_align",
                "ft.onto.base_ontology.Sentence": "auto_align",
            },
            "insertion_op_configs": {
                "type": "forte.processors.data_augment.algorithms."
                "dictionary_replacement_op.DictionaryReplacementOp",
                "kwargs": {
                    "dictionary_class": (
                        "forte.processors.data_augment."
                        "algorithms.dictionary.WordnetDictionary"
                    ),
                    "prob": 1.0,
                    "lang": "eng",
                },
            },
            "alpha": 0.1,
            "stopwords": english_stopwords,
        }


class RandomDeletionDataAugmentOp(BaseDataAugmentationOp):
    r"""
    Data augmentation operation for the Random Insertion operation.
    Randomly remove each word in the sentence with probability alpha.
    """

    def augment(self, data_pack: DataPack) -> bool:
        anno: Annotation
        for anno in data_pack.get(self.configs["augment_entry"]):
            if random.random() < self.configs["alpha"]:
                try:
                    self.delete_annotation(anno)
                except ValueError:
                    return False
        return True

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
            Additional keys for Random Deletion:

            - augment_entry (str):
                Defines the entry the processor will augment.
                It should be a full qualified name of the entry class.
                For example, "ft.onto.base_ontology.Sentence".
                Default Value is 0.1

            - alpha:
                0 <= alpha <= 1. The probability to delete each word.
        """
        return {
            "augment_entry": "ft.onto.base_ontology.Token",
            "other_entry_policy": {
                "ft.onto.base_ontology.Document": "auto_align",
                "ft.onto.base_ontology.Sentence": "auto_align",
            },
            "alpha": 0.1,
        }
