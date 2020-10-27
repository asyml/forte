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
from typing import List, Tuple, Dict
from bisect import bisect_right
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Annotation, MultiPackLink, Link
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
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
        r"""
        The replaced_spans records the entries replaced by new texts.
        It is a map from datapack id to a list of tuples
        (entry, new text) inserted by :func: replace.
        The new text will be used for building new data pack.

        The data_pack_map maintains a mapping of the pack ids
        from the original to the augmented one. It is used when
        copying the MultiPackLink.

        The anno_maps keeps a map for each datapack id to track the
        annotation ids before and after the auto align. It maps the
        original annotation tid to the new annotation tid.

        The keys for all the three dicts are pack ids.
        """
        super().__init__()
        self.replaced_spans: Dict[int, List[Tuple[Annotation, str]]] = {}
        self.data_pack_map: Dict[int, int] = {}
        self.anno_maps: Dict[int, Dict[int, int]] = {}

    def _is_span_overlap(self, pid: int, begin: int, end: int) -> bool:
        r"""
        This function will check whether the new span
        has an overlap with any existing spans.
        Args:
            - pid: Datapack Id.
            - begin: The span begin index.
            - end: The span end index.
        Returns:
            True if the input span overlaps with
            any existing spans, False otherwise.
        """
        if pid not in self.replaced_spans:
            return False
        for span, _ in self.replaced_spans[pid]:
            if not(span.begin >= end or span.end <= begin):
                return True
        return False

    def replace(
            self,
            replacement_op: TextReplacementOp,
            input: Annotation
    ) -> bool:
        r"""
        This is a wrapper function to call the replacement op. After
        getting the augmented text, it will register the input & output
        for later batch process of building the new data pack.

        Args:
            - replacement_op: The class for data augmentation algorithm.
            - input: The entry to be replaced.
        Returns:
            True if the replaced span does not overlap with any existing spans,
            False otherwise.
        """
        # Ignore the new annotation if overlap.
        pid: int = input.pack.meta.pack_id
        if self._is_span_overlap(pid, input.begin, input.end):
            return False
        replaced_text: str = replacement_op.replace(input)
        if pid not in self.replaced_spans:
            self.replaced_spans[pid] = []
        self.replaced_spans[pid].append((input, replaced_text))
        return True

    def _get_replaced_spans(self, pid: int) -> List[Tuple[Annotation, str]]:
        r"""
        This function get the replaced spans and their new text.
        Args:
            pid: Datapack id.
        Returns:
            A list of the tuples (replaced entry, new text) for the data pack.
        """
        if pid not in self.replaced_spans:
            return []
        return self.replaced_spans[pid]

    def auto_align_annotations(
        self,
        data_pack: DataPack,
        replaced_annotations: List[Tuple[Annotation, str]],
    ) -> DataPack:
        r"""
        Function to replace some annotations with new strings.
        It will update the text and auto-align the annotation spans.
        The links are also copied if its parent & child are
        both present in the new pack.

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

        anno_map: Dict[int, int] = {}
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
                new_anno = create_class_with_kwargs(
                    entry,
                    {
                        "pack": new_pack,
                        "begin": span_new_begin,
                        "end": span_new_end
                    }
                )
                new_pack.add_entry(new_anno)
                anno_map[orig_anno.tid] = new_anno.tid

        # Iterate over and copy the links in the datapack.
        for link in data_pack.get(Link):
            parent: Entry = link.parent
            child: Entry = link.child
            if parent.tid not in anno_map or child.tid not in anno_map:
                continue
            new_parent: Entry = new_pack.get_entry(anno_map[parent.tid])
            new_child: Entry = new_pack.get_entry(anno_map[child.tid])
            new_link = type(link)(
                new_pack, new_parent, new_child)  # type: ignore
            new_pack.add_entry(new_link)

        pid = data_pack.meta.pack_id
        self.data_pack_map[pid] = new_pack.meta.pack_id
        self.anno_maps[pid] = anno_map
        return new_pack

    def _clear_states(self):
        r"""
        This function clears the states. It should be
        called after processing a multipack.
        """
        self.replaced_spans.clear()
        self.data_pack_map.clear()
        self.anno_maps.clear()

    def _copy_multi_pack_links(self, multi_pack: MultiPack):
        r"""
        This function copies the MultiPackLinks in the multipack.
        It could be used in tasks such as text generation, where
        MultiPackLink is used to align the source and target.

        Args:
            - input: The input multi pack.
        """
        for mpl in multi_pack.get(MultiPackLink):
            # Get the original Entry and DataPack.
            parent: Entry = mpl.get_parent()
            child: Entry = mpl.get_child()
            parent_pack: DataPack = parent.pack
            child_pack: DataPack = child.pack
            parent_pack_pid: int = parent_pack.meta.pack_id
            child_pack_pid: int = child_pack.meta.pack_id

            if parent_pack_pid not in self.data_pack_map \
                    or child_pack_pid not in self.data_pack_map \
                    or parent_pack_pid not in self.anno_maps \
                    or child_pack_pid not in self.anno_maps:
                continue
            # Get the new Entry and DataPack.

            new_parent_pack: DataPack = multi_pack.get_pack_at(
                multi_pack.get_pack_index(self.data_pack_map[parent_pack_pid])
            )
            new_child_pack: DataPack = multi_pack.get_pack_at(
                multi_pack.get_pack_index(self.data_pack_map[child_pack_pid])
            )
            new_parent_tid: int = self.anno_maps[parent_pack_pid][parent.tid]
            new_child_tid: int = self.anno_maps[child_pack_pid][child.tid]
            new_parent: Entry = new_parent_pack.get_entry(new_parent_tid)
            new_child: Entry = new_child_pack.get_entry(new_child_tid)
            # Copy the MultiPackLink.
            multi_pack.add_entry(
                MultiPackLink(
                    multi_pack, new_parent, new_child
                )
            )
        self._clear_states()

    def _process(self, input_pack: MultiPack):
        replacement_op = create_class_with_kwargs(
            self.configs["replacement_op"],
            class_args={
                "configs": self.configs['replacement_op_config']
            }
        )
        augment_entry = get_class(self.configs["augment_entry"])
        new_packs: List[Tuple[str, DataPack]] = []

        for pack_name, data_pack in input_pack.iter_packs():
            for anno in data_pack.get(augment_entry):
                self.replace(replacement_op, anno)
            new_pack_name = "augmented_" + pack_name
            new_pack = self.auto_align_annotations(
                data_pack=data_pack,
                replaced_annotations=self._get_replaced_spans(
                    data_pack.meta.pack_id
                )
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
            'other_entry_policy': {},
            'replacement_op': "",
            'replacement_op_config': {}
        })
        return config
