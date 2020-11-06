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
replacement ops to generate texts similar to those in the input pack
and create a new pack with them.
"""
from typing import List, Tuple
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Annotation
from forte.data.data_pack import DataPack
from forte.processors.base.base_processor import BaseProcessor
from forte.processors.data_augment.algorithms.text_replacement_op \
    import TextReplacementOp

__all__ = [
    "BaseDataAugmentProcessor",
    "ReplacementDataAugmentProcessor"
]


class BaseDataAugmentProcessor(BaseProcessor):
    r"""The base class of processors that augment data.
    This processor instantiates replacement ops where specific
    data augmentation algorithms are implemented. The replacement ops
    will run the algorithms and the processor will create Forte
    data structures based on the augmented inputs.
    """


class ReplacementDataAugmentProcessor(BaseDataAugmentProcessor):
    r"""
    Most of the Data Augmentation(DA) methods can be
    considered as replacement-based methods with different
    levels: character, word, sentence or document.
    """
    def __init__(self):
        """
        The replaced_spans records the entries replaced by new texts.
        It consists of tuples (entry, new text) inserted by :func: replace.
        The new text will be used for building new data pack.
        """
        super().__init__()
        self.replaced_spans: List[Tuple[Entry, str]] = []

    def replace(self, replacement_op: TextReplacementOp, input: Entry):
        """
        This is a wrapper function to call the replacement op. After
        getting the augmented text, it will register the input & output
        for later batch process of building the new data pack.
        """
        replaced_text: str
        is_replace: bool
        is_replace, replaced_text = replacement_op.replace(input)
        if is_replace:
            self.replaced_spans.append((input, replaced_text))

    def auto_align_annotations(
        self,
        data_pack: DataPack,
        replaced_annotations: List[Tuple[Annotation, str]]
    ) -> DataPack:
        r"""
        Function to replace some annotations with new strings.
        It will update the text and auto-align the annotation spans.

        Args:
            data_pack: Datapack holding the annotations to be replaced.
            replaced_annotations: A list of tuples(annotation, new string).
            The text for annotation will be updated with the new string.

        Returns:
            A new data_pack holds the text after replacement. The annotations
            in the original data pack will be copied and auto-aligned as
            instructed by the "other_entry_policy".

        """
        pass

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - augment_entries: defines the entries the processor
            will augment. It should be a full path to the entry class.
            - other_entry_policy: a dict specifying the policies for
            other entries.

            If "auto_align", the span of the entry will be automatically
            modified according to its original location. However, some
            spans might become invalid after the augmentation, for
            example, the tokens within a replaced sentence may disappear.

            Entries not in the dict will not be copied to the new data pack.

            Example: {
                "ft.onto.base_ontology.Document": "auto_align",
                "ft.onto.base_ontology.Sentence": "auto_align"
            }
        """
        config = super().default_configs()
        config.update({
            'augment_entry': "ft.onto.base_ontology.Sentence",
            'other_entry_policy': {}
        })
        return config
