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
from collections import defaultdict
from typing import List, Tuple, Dict, DefaultDict, Set
from bisect import bisect_right
from forte.data.ontology.core import Entry
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.ontology.top import Annotation, MultiPackLink, Link
from forte.data.span import Span
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
        - replaced_annos: {datapack id: List[span, text]}
            It records the spans replaced by new texts.
            It is a map from datapack id to a list of tuples
            (span, new text) inserted by :func: replace.
            The new text will be used for building new data pack.

        - inserted_annos: {datapack id: Dict{position: length}}
            It records the inserted spans, mapping from datapack id
            to a dictionary (position -> length) inserted
            by :func: insert.
            The position is the index in the original datapack
            of insertion, and the length is the length of the
            inserted string.

        - deleted_annos: {datapack id: Set[annotation tid]}
            It records the deleted spans, mapping from datapack id
            to a set of annotation tids appended by :func: delete.

        - data_pack_map: {orig pack id: new pack id}
            It maintains a mapping from the pack id
            of the original pack to the pack id of augmented pack.
            It is used when copying the MultiPackLink.

        - anno_maps: {datapack id: Dict{orig tid, new tid}}
            It is a map for tracking the annotation ids
            before and after the auto align. It maps the
            original annotation tid to the new annotation tid.
            It is used when copying the MultiPackLink.
        """
        super().__init__()
        self.replaced_annos: DefaultDict[int, List[Tuple[Span, str]]] = \
            defaultdict(list)
        self.inserted_annos: DefaultDict[int, Dict[int, int]] = \
            defaultdict(dict)
        self.deleted_annos: DefaultDict[int, Set[int]] = defaultdict(set)

        self.data_pack_map: Dict[int, int] = {}
        self.anno_maps: Dict[int, Dict[int, int]] = {}
        self.other_entry_policy: Dict[str, str] = {}

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        # Parse the "other_entry_policy" in the configuration.
        self.other_entry_policy = self._get_other_entry_policy_dict()

    def _get_other_entry_policy_dict(self) -> Dict[str, str]:
        r"""
        A helper function for parsing the "other_entry_policy"
        configuration.
        """
        entries = self.configs['other_entry_policy']["entry"]
        policies = self.configs['other_entry_policy']["policy"]
        other_entry_policy = {}
        for entry, policy in zip(entries, policies):
            other_entry_policy[entry] = policy
        return other_entry_policy

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
        for span, _ in self.replaced_annos[pid]:
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

        It will ignore the input if it has an overlap with the already
        augmented spans.

        Args:
            - replacement_op: The class for data augmentation algorithm.
            - input: The entry to be replaced.
        Returns:
            A bool value. True if the replacement happened, False otherwise.
        """
        # Ignore the new annotation if overlap.
        pid: int = input.pack.meta.pack_id
        if self._is_span_overlap(pid, input.begin, input.end):
            return False
        replaced_text: str
        is_replace: bool
        is_replace, replaced_text = replacement_op.replace(input)
        if is_replace:
            self.replaced_annos[pid].append((input.span, replaced_text))
            return True
        return False

    def insert(
            self,
            inserted_text: str,
            data_pack: DataPack,
            pos: int
    ) -> bool:
        r"""
        This is a wrapper function to insert a new annotation. After
        getting the inserted text, it will register the input & output
        for later batch process of building the new data pack.

        The insertion at each position can only occur once. If there
        is already an insertion at current position, it will abort.
        Args:
            - replacement_op: The class for data augmentation algorithm.
                For insertion, it should be able to generate a string
                with an empty input.
            - data_pack: The datapack for insertion.
            - pos: The position(index) of insertion.
        Returns:
            A bool value. True if the insertion happened, False otherwise.
        """
        pid: int = data_pack.meta.pack_id
        if self._is_span_overlap(pid, pos, pos):
            return False

        if pos not in self.inserted_annos[pid]:
            self.replaced_annos[pid].append((Span(pos, pos), inserted_text))
            self.inserted_annos[pid][pos] = len(inserted_text)
            return True
        return False

    def delete(
            self,
            input: Annotation
    ) -> bool:
        r"""
        This is a wrapper function to delete an annotation.

        Args:
            -input: The annotation to remove.
        Returns:
            A bool value. True if the deletion happened, False otherwise.
        """
        pid: int = input.pack.meta.pack_id
        self.replaced_annos[pid].append((input.span, ""))
        self.deleted_annos[pid].add(input.tid)
        return True

    def auto_align_annotations(
        self,
        data_pack: DataPack,
        replaced_annotations: List[Tuple[Span, str]],

    ) -> DataPack:
        r"""
        Function to replace some annotations with new strings.
        It will copy and update the text of datapack and
        auto-align the annotation spans.

        The links are also copied if its parent & child are
        both present in the new pack.

        Args:
            - data_pack: The Datapack holding the replaced annotations.
            - replaced_annotations: A list of tuples(span, new string).
            The text and span of the annotations will be updated
            with the new string.

        Returns:
            A new data_pack holds the text after replacement. The annotations
            in the original data pack will be copied and auto-aligned as
            instructed by the "other_entry_policy" in the configuration.
        """
        if len(replaced_annotations) == 0:
            return deepcopy(data_pack)

        # Sort the annotations by span beginning index.
        replaced_annotations = sorted(
            replaced_annotations,
            key=lambda x: (x[0].begin, x[0].end)
        )
        spans: List[Tuple[int, int]] = [
            (span.begin, span.end) for span, _ in replaced_annotations]
        replacement_strs: List[str] = [
            replacement_str for _, replacement_str in replaced_annotations]

        # Get the new text for the new data pack.
        new_text: str = ""
        for i, span in enumerate(spans):
            new_span_str = replacement_strs[i]
            # First, get the gap text between last and this span.
            last_span_end: int = spans[i - 1][1] if i > 0 else 0
            gap_text: str = data_pack.text[last_span_end: span[0]]
            new_text += gap_text
            # Then, append the replaced new text.
            new_text += new_span_str
        # Finally, append to new_text the text after the last span.
        new_text += data_pack.text[spans[-1][1]:]

        # Get the span (begin, end) before and after replacement.
        new_spans: List[Tuple[int, int]] = []

        # Bias is the delta between the beginning
        # indices before & after replacement.
        bias: int = 0
        for i, span in enumerate(spans):
            old_begin: int = spans[i][0]
            old_end: int = spans[i][1]
            new_begin: int = old_begin + bias
            new_end = new_begin + len(replacement_strs[i])
            new_spans.append((new_begin, new_end))
            bias = new_end - old_end

        def modify_index(
                index: int,
                old_spans: List[Tuple[int, int]],
                new_spans: List[Tuple[int, int]],
                is_begin: bool,
                is_inclusive: bool
        ) -> int:
            r"""
            A helper function to map an index before replacement
            to the index after replacement. The old spans and
            new spans are anchor indices for the mapping.

            Args:
                - index: The index to map.
                - old_spans: The spans before replacement.
                - new_spans: The spans after replacement.
                - is_begin: True if the index is a span begin index.
                - is_inclusive: True if the span constructed by the aligned
                    index should include inserted spans.
            Returns:
                The aligned index.

            If the old spans are:
                [0, 1], [2, 3], [4, 6]
            the new spans are:
                [0, 4], [5, 7], [8, 9]
            the input index is:
                3
            and we ignore the insertions,
            the algorithm will first locate the last span with
            a begin index less or equal than the target index,
            ([2,3]), and find the corresponding span in new spans([5,7]).
            Then we calculate the delta index(7-3=4) and update our
            input index(3+4=7). The output then is 7.

            When insertion is considered, there will be spans
            with the same begin index, for example,
            [0, 1], [1, 1], [1, 2]. The output will depend on
            whether to include the inserted span, and whether the
            input index is a begin or an end index.

            If the old spans are:
                [0, 1], [1, 1], [1, 2]
            the new spans are:
                [0, 2], [2, 4], [4, 5]
            the input index is:
                1,
            the output will be 2 if both is_inclusive and is_begin are True,
            because the inserted [1, 1] should be included in the span.
            If the is_inclusive=True but is_begin=False, the output will be
            4 because the index is an end index of the span.
            """

            # Get the max index for binary search.
            max_index: int = old_spans[-1][1] + 1

            last_span_ind: int = bisect_right(
                old_spans, (index, max_index)
            ) - 1

            # If there is an inserted span, it will
            # always be the first of those spans with
            # the same begin index.
            if last_span_ind >= 0:
                if is_inclusive:
                    if is_begin:
                        # When inclusive, move the begin index
                        # to the left to include the inserted span.
                        if last_span_ind > 0 and \
                                old_spans[last_span_ind - 1][0] == index:
                            # Old spans: [0, 1], [1, 1], [1, 3]
                            # Target index: 1
                            # Change last_span_index from 2 to 1
                            # to include the [1, 1] span.
                            last_span_ind -= 1
                        else:
                            # Old spans: [0, 1], [1, 1], [2, 3]
                            # Target index: 1
                            # last_span_index: 1
                            # No need to change.
                            pass

                else:
                    if not is_begin:
                        # When exclusive, move the end index
                        # to the left to exclude the inserted span.
                        if last_span_ind > 0 and \
                                old_spans[last_span_ind - 1][0] == index:
                            # Old spans: [0, 1], [1, 1], [1, 3]
                            # Target index: 1
                            # Change last_span_index from 2 to 0
                            # to exclude the [1, 1] span.
                            last_span_ind -= 2
                        elif old_spans[last_span_ind][0] == index and \
                            old_spans[last_span_ind][1] == index:
                            # Old spans: [0, 1], [1, 1], [2, 3]
                            # Target index: 1
                            # Change last_span_index from 1 to 0
                            # to exclude the [1, 1] span.
                            last_span_ind -= 1

            if last_span_ind < 0:
                # There is no replacement before this index.
                return index
            # Find the nearest anchor point on the left of current index.
            # Start from the span's begin index.
            delta_index: int = new_spans[last_span_ind][0] - \
                               old_spans[last_span_ind][0]

            if old_spans[last_span_ind][0] == old_spans[last_span_ind][1] \
                    and old_spans[last_span_ind][0] == index \
                    and is_begin \
                    and is_inclusive:
                return index + delta_index

            if old_spans[last_span_ind][1] <= index:
                # Use the span's end index as anchor, if possible.
                delta_index = new_spans[last_span_ind][1] - \
                              old_spans[last_span_ind][1]
            return index + delta_index

        new_pack: DataPack = DataPack()
        new_pack.set_text(new_text)

        entries_to_copy: List[str] = \
            list(self.other_entry_policy.keys()) + \
            [self.configs['augment_entry']]

        anno_map: Dict[int, int] = {}

        insert_ind: int = 0
        pid: int = data_pack.meta.pack_id

        inserted_annos: List[Tuple[int, int]] = sorted(
            list(self.inserted_annos[pid].items()),
            key=lambda x: x[0]
        )

        def _insert_new_span(
                insert_ind: int,
                inserted_annos: List[Tuple[int, int]],
                new_pack: DataPack,
                spans: List[Tuple[int, int]],
                new_spans: List[Tuple[int, int]]
        ):
            r"""
            An internal helper function for insertion.
            """
            pos: int
            length: int
            pos, length = inserted_annos[insert_ind]
            insert_end: int = modify_index(
                pos,
                spans,
                new_spans,
                is_begin=False,
                # Include the inserted span itself.
                is_inclusive=True
            )
            insert_begin: int = insert_end - length
            new_anno = create_class_with_kwargs(
                entry,
                {
                    "pack": new_pack,
                    "begin": insert_begin,
                    "end": insert_end
                }
            )
            new_pack.add_entry(new_anno)

        # Iterate over all the original entries and modify their spans.
        for entry in entries_to_copy:
            for orig_anno in data_pack.get(get_class(entry)):
                # Dealing with insertion/deletion only for augment_entry.
                if entry == self.configs['augment_entry']:
                    while insert_ind < len(inserted_annos) and \
                            inserted_annos[insert_ind][0] <= orig_anno.begin:
                        # Preserve the order of the spans with merging sort.
                        # It is a 2-way merging from the inserted spans
                        # and original spans based on the begin index.
                        _insert_new_span(
                            insert_ind,
                            inserted_annos,
                            new_pack,
                            spans,
                            new_spans
                        )
                        insert_ind += 1

                    # Deletion
                    if orig_anno.tid in self.deleted_annos[pid]:
                        continue

                # Auto align the spans.
                span_new_begin: int = orig_anno.begin
                span_new_end: int = orig_anno.end

                if entry == self.configs['augment_entry'] \
                        or self.other_entry_policy[entry] \
                        == 'auto_align':
                    # Only inclusive when the entry is not augmented.
                    # E.g.: A Sentence include the inserted Token on the edge.
                    # E.g.: A Token shouldn't include a nearby inserted Token.
                    is_inclusive = entry != self.configs['augment_entry']
                    span_new_begin = modify_index(
                        orig_anno.begin, spans, new_spans, True, is_inclusive)
                    span_new_end = modify_index(
                        orig_anno.end, spans, new_spans, False, is_inclusive)

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

            # Deal with spans after the last annotation in the original pack.
            if entry == self.configs['augment_entry']:
                while insert_ind < len(inserted_annos):
                    _insert_new_span(
                        insert_ind,
                        inserted_annos,
                        new_pack,
                        spans,
                        new_spans
                    )
                    insert_ind += 1

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

        self.data_pack_map[pid] = new_pack.meta.pack_id
        self.anno_maps[pid] = anno_map
        return new_pack

    def _clear_states(self):
        r"""
        This function clears the states. It should be
        called after processing a multipack.
        """
        self.replaced_annos.clear()
        self.inserted_annos.clear()
        self.deleted_annos.clear()
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

            # Only copy the link when both packs are present.
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

            # Only copy the link when both entries are present.
            if parent.tid not in self.anno_maps[parent_pack_pid] \
                    or child.tid not in self.anno_maps[child_pack_pid]:
                continue
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
            self.configs["kwargs"]["data_aug_op"],
            class_args={
                "configs": self.configs["kwargs"]["data_aug_op_config"]
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
        Following are the keys for this dictionary:
            - augment_entries: defines the entries the processor
            will augment. It should be a full path to the entry class.
            - other_entry_policy: a dict specifying the policies for
            other entries. It has two keys: "entry" and "policy".

            The values of "policy" are aligned with the values of "entry".
            They are two aligned lists. The "entry" should be class name.
            The "policy" specifies how to process the corresponding entries
            after replacement.

            If the policy is "auto_align", the span of the entry
            will be automatically modified according to its original location.
            However, some spans might become invalid after the augmentation,
            for example, the tokens within a replaced sentence may disappear.

            Entries not in the "entry" will not be copied to the new data pack.

            Example:
            {
                "entry": [
                    "ft.onto.base_ontology.Document",
                    "ft.onto.base_ontology.Sentence"
                ],
                "policy": ["auto_align", "auto_align"]
            }
        """
        config = super().default_configs()
        config.update({
            'augment_entry': "ft.onto.base_ontology.Sentence",
            'other_entry_policy': {
                "entry": [],
                "policy": [],
            },
            "type": "data_augmentation_op",
            "kwargs": {
                'data_aug_op': "",
                'data_aug_op_config': {}
            }
        })
        return config
