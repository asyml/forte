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
from copy import deepcopy
from typing import List, Tuple
from bisect import bisect_right
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Annotation
from forte.data.data_pack import DataPack
from forte.processors.base.base_processor import BaseProcessor
from forte.utils.utils import get_class, create_class_with_kwargs
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

    def _is_span_overlap(self, begin, end):
        r"""
        This function will check whether the new span
        has an overlap with any existing spans.
        """
        for span in self.replaced_spans:
            if span.begin >= end or span.end <= begin:
                return True
        return False

    def replace(self, replacement_op: TextReplacementOp, input: Entry) -> bool:
        """
        This is a wrapper function to call the replacement op. After
        getting the augmented text, it will register the input & output
        for later batch process of building the new data pack.
        """
        # Ignore the new annotation if overlap.
        if self._is_span_overlap(input.begin, input.end):
            return False
        replaced_text: str = replacement_op.replace(input)
        self.replaced_spans.append((input, replaced_text))
        return True

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
        if len(replaced_annotations) == 0:
            return deepcopy(data_pack)

        # Sort the annotations by span beginning index.
        replaced_annotations = sorted(
            replaced_annotations,
            key=lambda x: x[0].begin
        )
        annotations: List[Annotation] = [
            annotation for annotation, _ in replaced_annotations]
        replacement_strs: List[str] = [
            replacement_str for _, replacement_str in replaced_annotations]

        # Get the new text for the new data pack.
        new_text: str = ""
        for i, anno in enumerate(annotations):
            new_anno_str = replacement_strs[i]
            # First, get the gap text between last and this annotation.
            last_anno_end: int = annotations[i - 1].end if i > 0 else 0
            gap_text: str = data_pack.text[last_anno_end: anno.begin]
            new_text += gap_text
            # Then, append the replaced new text.
            new_text += new_anno_str
        # Finally, append to new_text the text after the last annotation.
        new_text += data_pack.text[annotations[-1].end:]

        # Get the span (begin, end) before and after replacement.
        old_spans: List[Tuple[int, int]] = [
            (anno.begin, anno.end) for anno in annotations]
        new_spans: List[Tuple[int, int]] = []

        # Bias is the delta of beginning index of two spans.
        bias: int = 0
        for i in range(len(annotations)):
            old_begin: int = old_spans[i][0]
            old_end: int = old_spans[i][1]
            new_begin: int = old_begin + bias
            new_end = new_begin + len(replacement_strs[i])
            new_spans.append((new_begin, new_end))
            bias = new_end - old_end

        def modify_index(index: int, old_spans: List[Tuple[int, int]],
                         new_spans: List[Tuple[int, int]]):
            r"""
            A helper function to map an index before replacement
            to the index after replacement. The old spans and
            new spans are anchor indices for the mapping.
            """

            # Get the max index for binary search.
            max_index: int = old_spans[-1][1] + 1
            last_span_ind: int = bisect_right(
                old_spans, (index, max_index)
            ) - 1
            if last_span_ind < 0:
                # There is no replacement before this index.
                return index

            # Find the nearest anchor point on the left of current index.
            # Start from the span's begin index.
            delta_index: int = new_spans[last_span_ind][0] - \
                               old_spans[last_span_ind][0]
            if old_spans[last_span_ind][1] <= index:
                # Use the span's end index as anchor, if possible.
                delta_index = new_spans[last_span_ind][1] - \
                              old_spans[last_span_ind][1]
            return index + delta_index

        new_pack: DataPack = DataPack()
        new_pack.set_text(new_text)

        entries_to_copy: List[str] = \
            list(self.configs['auto_align_entries'].keys()) + \
            [self.configs['augment_entry']]

        # Iterate over all the original entries and modify their spans.
        for entry in entries_to_copy:
            for orig_anno in data_pack.get(get_class(entry)):
                # Auto align the spans.
                span_new_begin: int = orig_anno.begin
                span_new_end: int = orig_anno.end

                if entry == self.configs['augment_entry'] \
                        or self.configs['auto_align_entries'][entry] \
                        == 'auto_align':
                    span_new_begin = modify_index(
                        orig_anno.begin, old_spans, new_spans)
                    span_new_end = modify_index(
                        orig_anno.end, old_spans, new_spans)
                new_pack.add_entry(
                    create_class_with_kwargs(
                        entry,
                        {
                            "pack": new_pack,
                            "begin": span_new_begin,
                            "end": span_new_end
                        }
                    )
                )
        return new_pack

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
