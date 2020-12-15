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
from typing import List, Tuple, Dict, DefaultDict, Set, Union, cast
from bisect import bisect_right
from sortedcontainers import SortedList, SortedDict
from forte.data.ontology.core import Entry, BaseLink
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.ontology.top import (
    Annotation, MultiPackLink, Link, MultiPackGroup, Group
)
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


def modify_index(
        index: int,
        # Both of the following spans should be SortedList.
        # Use List to avoid typing errors.
        old_spans: List[Tuple[int, int]],
        new_spans: List[Tuple[int, int]],
        is_begin: bool,
        is_inclusive: bool
) -> int:
    r"""
    A helper function to map an index before replacement
    to the index after replacement.

    An index is the character offset in the data pack.
    The old_spans are the inputs of replacement, and the new_spans
    are the outputs. Each of the span is a tuple with start and end
    index. The old_spans and new_spans are anchors for the mapping,
    because we depend on them to determine the position change of the
    index.

    Given an index, the function will find its nearest old span, and
    calculate the difference between the position of the old
    span and its corresponding new span. The position change is then
    applied to the input index.

    Args:
        index (int): The index to map.
        old_spans (List): The spans before replacement.
        new_spans (List): The spans after replacement.
        is_begin (bool): True if the index is a span begin index.
        is_inclusive (bool): True if the span constructed by the aligned
            index should include inserted spans.
    Returns:
        The aligned index.

    If the old spans are [0, 1], [2, 3], [4, 6],
    the new spans are [0, 4], [5, 7], [8, 9],
    the input index is 3, and there are no insertions,
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

    If the old spans are [0, 1], [1, 1], [1, 2],
    the new spans are [0, 2], [2, 4], [4, 5],
    the input index is 1, the output will be 2 if both
    is_inclusive and is_begin are True,
    because the inserted [1, 1] should be included in the span.
    If the is_inclusive=True but is_begin=False, the output will be
    4 because the index is an end index of the span.
    """

    # Get the max index for binary search.
    max_index: int = old_spans[-1][1] + 1

    # This is the last span that has a start index less than
    # the input index. The position change of this span determines
    # the modification we will apply to the input index.
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
        super().__init__()
        # :attr:`_replaced_annos`: {datapack id: SortedList[span, new text]}
        # It records the spans replaced by new texts.
        # It is a map from datapack id to a list of tuples
        # (span, new text) inserted by :func:`replace`.
        # The new text will be used for building new data pack.
        self._replaced_annos: DefaultDict[int, SortedList[Tuple[Span, str]]] = \
            defaultdict(
                lambda: SortedList([], key=lambda x: (x[0].begin, x[0].end))
            )

        # :attr:`_inserted_annos_pos_len`: {datapack id: Dict{position: length}}
        # It records the position and length of inserted spans,
        # mapping from datapack id to a dictionary (position -> length)
        # inserted by :func:`insert`.
        # The position is the index in the original datapack
        # of insertion, and the length is the length of the inserted string.
        self._inserted_annos_pos_len: DefaultDict[int, SortedDict[int, int]] = \
            defaultdict(
                lambda: SortedDict({})
            )

        # :attr:`_deleted_annos_id`: {datapack id: Set[annotation tid]}
        # It records the deleted span ids, mapping from datapack id
        # to a set of annotation tids appended by :func:`delete`.
        self._deleted_annos_id: DefaultDict[int, Set[int]] = defaultdict(set)

        # :attr:`_data_pack_map`: {orig pack id: new pack id}
        # It maintains a mapping from the pack id
        # of the original pack to the pack id of augmented pack.
        # It is used when copying the MultiPackLink and MultiPackGroup.
        self._data_pack_map: Dict[int, int] = {}

        # :attr:`_entry_maps`: {datapack id: Dict{orig tid, new tid}}
        # It is a map for tracking the annotation ids
        # before and after the auto align. It maps the
        # original annotation tid to the new annotation tid.
        # It is used when copying the Link/Group/MultiPackLink/MultiPackGroup.
        self._entry_maps: Dict[int, Dict[int, int]] = {}
        self._other_entry_policy: Dict[str, str] = {}

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self._other_entry_policy = self.configs["other_entry_policy"]["kwargs"]

    def _overlap_with_existing(self, pid: int, begin: int, end: int) -> bool:
        r"""
        This function will check whether the new span
        has an overlap with any existing spans.
        Args:
            pid: Datapack Id.
            begin: The span begin index.
            end: The span end index.
        Returns:
            True if the input span overlaps with
            any existing spans, False otherwise.
        """
        for span, _ in self._replaced_annos[pid]:
            if not(span.begin >= end or span.end <= begin):
                return True
        return False

    def _replace(
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
            replacement_op: The class for data augmentation algorithm.
            input: The entry to be replaced.
        Returns:
            A bool value. True if the replacement happened, False otherwise.
        """
        # Ignore the new annotation if overlap.
        pid: int = input.pack.meta.pack_id
        if self._overlap_with_existing(pid, input.begin, input.end):
            return False
        replaced_text: str
        is_replace: bool
        is_replace, replaced_text = replacement_op.replace(input)
        if is_replace:
            self._replaced_annos[pid].add((input.span, replaced_text))
            return True
        return False

    def _insert(
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
        is already an insertion at current position, it will abort the
        insertion and return False.
        Args:
            inserted_text: The text string to insert.
            data_pack: The datapack for insertion.
            pos: The position(index) of insertion.
        Returns:
            A bool value. True if the insertion happened, False otherwise.
        """
        pid: int = data_pack.meta.pack_id
        if self._overlap_with_existing(pid, pos, pos):
            return False

        if pos not in self._inserted_annos_pos_len[pid]:
            self._replaced_annos[pid].add((Span(pos, pos), inserted_text))
            self._inserted_annos_pos_len[pid][pos] = len(inserted_text)
            return True
        return False

    def _delete(
            self,
            input: Annotation
    ) -> bool:
        r"""
        This is a wrapper function to delete an annotation.

        Args:
            input: The annotation to remove.
        Returns:
            A bool value. True if the deletion happened, False otherwise.
        """
        pid: int = input.pack.meta.pack_id
        self._replaced_annos[pid].add((input.span, ""))
        self._deleted_annos_id[pid].add(input.tid)
        return True

    def _auto_align_annotations(
        self,
        data_pack: DataPack,
        replaced_annotations: SortedList,
    ) -> DataPack:
        r"""
        Function to replace some annotations with new strings.
        It will copy and update the text of datapack and
        auto-align the annotation spans.

        The links are also copied if its parent & child are
        both present in the new pack.

        The groups are copied if all its members are present
        in the new pack.

        Args:
            data_pack: The Datapack holding the replaced annotations.
            replaced_annotations: A SortedList of tuples(span, new string).
                The text and span of the annotations will be updated
                with the new string.

        Returns:
            A new data_pack holds the text after replacement. The annotations
            in the original data pack will be copied and auto-aligned as
            instructed by the "other_entry_policy" in the configuration.
            The links and groups will be copied if there members are copied.
        """
        if len(replaced_annotations) == 0:
            return deepcopy(data_pack)

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

        new_pack: DataPack = DataPack()
        new_pack.set_text(new_text)

        entries_to_copy: List[str] = \
            list(self._other_entry_policy.keys()) + \
            [self.configs['augment_entry']]

        entry_map: Dict[int, int] = {}
        insert_ind: int = 0
        pid: int = data_pack.meta.pack_id

        inserted_annos: List[Tuple[int, int]] = list(
            self._inserted_annos_pos_len[pid].items()
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
                    if orig_anno.tid in self._deleted_annos_id[pid]:
                        continue

                # Auto align the spans.
                span_new_begin: int = orig_anno.begin
                span_new_end: int = orig_anno.end

                if entry == self.configs['augment_entry'] \
                        or self._other_entry_policy[entry] \
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
                entry_map[orig_anno.tid] = new_anno.tid

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

        # Iterate over and copy the links/groups in the datapack.
        for link in data_pack.get(Link):
            self._copy_link_or_group(link, entry_map, new_pack)
        for group in data_pack.get(Group):
            self._copy_link_or_group(group, entry_map, new_pack)

        self._data_pack_map[pid] = new_pack.meta.pack_id
        self._entry_maps[pid] = entry_map
        return new_pack

    def _clear_states(self):
        r"""
        This function clears the states. It should be
        called after processing a multipack.
        """
        self._replaced_annos.clear()
        self._inserted_annos_pos_len.clear()
        self._deleted_annos_id.clear()
        self._data_pack_map.clear()
        self._entry_maps.clear()

    def _copy_link_or_group(
            self,
            entry: Union[Link, Group],
            entry_map: Dict[int, int],
            new_pack: DataPack,
    ) -> bool:
        r"""
        This function copies a Link/Group in the data pack.
        If the children Link/Group does not exist, it will recursively
        create the children Link/Group. If the children Annotation
        does not exist, it will abort and return False.

        Args:
            entry: The Link/Group in the original data pack to copy.
            entry_map: The dictionary mapping original entry to copied entry.
            new_pack: The new data pack, which is the destination of copy.
        Returns:
            A bool value indicating whether the copy happens.
        """

        # If the entry has been copied, return True.
        if entry.tid in entry_map:
            return True

        # The entry should be either Link or Group.
        is_link: bool = isinstance(entry, Link)

        # Get the children entries.
        children: List[Entry]
        if is_link:
            children = [entry.get_parent(), entry.get_child()]
        else:
            children = entry.get_members()

        # Copy the children entries.
        new_children: List[Entry] = []
        for child_entry in children:
            if isinstance(child_entry, Annotation):
                # Children Annotation must have been copied.
                if child_entry.tid not in entry_map:
                    return False
            elif isinstance(child_entry, (Link, Group)):
                # Recursively copy the children Links/Groups.
                if not self._copy_link_or_group(
                        child_entry, entry_map, new_pack):
                    return False
            else:
                return False
            new_child: Entry = new_pack.get_entry(
                entry_map[child_entry.tid]
            )
            new_children.append(new_child)

        # Create the new entry and add to the new pack.
        new_entry: Entry
        if is_link:
            entry = cast(Link, entry)
            new_link_parent: Entry = new_children[0]
            new_link_child: Entry = new_children[1]
            new_entry = type(entry)(
                new_pack, new_link_parent, new_link_child)  # type: ignore
        else:
            entry = cast(Group, entry)
            new_entry = type(entry)(
                new_pack, new_children)  # type: ignore
        new_pack.add_entry(new_entry)
        entry_map[entry.tid] = new_entry.tid
        return True

    def _copy_multi_pack_link_or_group(
            self,
            entry: Union[MultiPackLink, MultiPackGroup],
            multi_pack: MultiPack
    ) -> bool:
        r"""
        This function copies a MultiPackLink/MultiPackGroup in the multipack.
        It could be used in tasks such as text generation, where
        MultiPackLink is used to align the source and target.

        Args:
            entry: The MultiPackLink/MultiPackGroup to copy.
            multi_pack: The multi_pack contains the input entry.
        Returns:
            A bool value indicating whether the copy happens.
        """
        # The entry should be either MultiPackLink or MultiPackGroup.
        is_link: bool = isinstance(entry, BaseLink)
        children: List[Entry]
        if is_link:
            children = [entry.get_parent(), entry.get_child()]
        else:
            children = entry.get_members()

        # Get the copied children entries.
        new_children: List[Entry] = []
        for child_entry in children:
            child_pack: DataPack = child_entry.pack
            child_pack_pid: int = child_pack.meta.pack_id
            # The new pack should be present.
            if child_pack_pid not in self._data_pack_map \
                    or child_pack_pid not in self._entry_maps:
                return False
            new_child_pack: DataPack = multi_pack.get_pack_at(
                multi_pack.get_pack_index(self._data_pack_map[child_pack_pid])
            )
            # The new child entry should be present.
            if child_entry.tid not in self._entry_maps[child_pack_pid]:
                return False
            new_child_tid: int = \
                self._entry_maps[child_pack_pid][child_entry.tid]
            new_child_entry: Entry = new_child_pack.get_entry(new_child_tid)
            new_children.append(new_child_entry)

        # Create the new entry and add to the multi pack.
        new_entry: Entry
        if is_link:
            entry = cast(MultiPackLink, entry)
            new_link_parent: Entry = new_children[0]
            new_link_child: Entry = new_children[1]
            new_entry = type(entry)(
                multi_pack, new_link_parent, new_link_child)  # type: ignore
        else:
            entry = cast(MultiPackGroup, entry)
            new_entry = type(entry)(
                multi_pack, new_children)  # type: ignore
        multi_pack.add_entry(new_entry)
        return True

    def _augment(self, input_pack: MultiPack, aug_pack_names: List[str]):
        r"""
        This function calls the data augmentation ops and
        modifies the input in-place. This function only applies for
        replacement-based methods. The subclasses should override
        this function to implement other data augmentation methods, such
        as Easy Data Augmentation.

        Args:
            input_pack: The input MultiPack.
            aug_pack_names: The packs names for DataPacks to be augmented.
        """
        replacement_op = create_class_with_kwargs(
            self.configs["data_aug_op"],
            class_args={
                "configs": self.configs["data_aug_op_config"]["kwargs"]
            }
        )
        augment_entry = get_class(self.configs["augment_entry"])

        for pack_name in aug_pack_names:
            data_pack: DataPack = input_pack.get_pack(pack_name)
            for anno in data_pack.get(augment_entry):
                self._replace(replacement_op, anno)

    def _process(self, input_pack: MultiPack):
        # Get the pack names for augmentation.
        aug_pack_names: List[str] = []

        # Check if the DataPack exists.
        for pack_name in self.configs["augment_pack_names"]["kwargs"].keys():
            if pack_name in input_pack.pack_names:
                aug_pack_names.append(pack_name)

        if len(self.configs["augment_pack_names"]["kwargs"].keys()) == 0:
            # Augment all the DataPacks if not specified.
            aug_pack_names = list(input_pack.pack_names)

        self._augment(input_pack, aug_pack_names)
        new_packs: List[Tuple[str, DataPack]] = []
        for aug_pack_name in aug_pack_names:
            new_pack_name: str = \
                self.configs["augment_pack_names"]["kwargs"].get(
                    aug_pack_name, "augmented_" + aug_pack_name
                )
            data_pack = input_pack.get_pack(aug_pack_name)
            new_pack = self._auto_align_annotations(
                data_pack=data_pack,
                replaced_annotations=self._replaced_annos[
                    data_pack.meta.pack_id
                ]
            )
            new_packs.append((new_pack_name, new_pack))

        for new_pack_name, new_pack in new_packs:
            input_pack.add_pack_(new_pack, new_pack_name)

        # Copy the MultiPackLinks/MultiPackGroups
        for mpl in input_pack.get(MultiPackLink):
            self._copy_multi_pack_link_or_group(mpl, input_pack)
        for mpg in input_pack.get(MultiPackGroup):
            self._copy_multi_pack_link_or_group(mpg, input_pack)

        # Must be called after processing each multipack
        # to reset internal states.
        self._clear_states()

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
            - augment_entry:
                Defines the entry the processor will augment.
                It should be a full qualified name of the entry class.
                For example, "ft.onto.base_ontology.Sentence".
            - other_entry_policy:
                A dict specifying the policies for other entries.
                The key should be a full qualified class name.
                The policy(value of the dict) specifies how to process
                the corresponding entries after replacement.

                If the policy is "auto_align", the span of the entry
                will be automatically modified according to its original
                location. However, some spans might become invalid after
                the augmentation, for example, the tokens within a
                replaced sentence may disappear.

                Annotations not in the "other_entry_policy" will not
                be copied to the new data pack. The Links and Groups
                will be copied as well if the annotations they are
                attached to are copied.

                Example:
                    .. code-block:: python

                        'other_entry_policy': {
                            "kwargs": {
                                "ft.onto.base_ontology.Document": "auto_align",
                                "ft.onto.base_ontology.Sentence": "auto_align",
                            }
                        }

            - type:
                Should not modify this field, in order to use the kwargs.
            - data_aug_op:
                The data augmentation Op for the processor.
                It should be a full qualified class name.

                Example:
                    "forte.processors.data_augment.algorithms.
                    text_replacement_op.TextReplacementOp"
            - data_aug_op_config:
                The configuration for data augmentation Op.

                Example:
                    .. code-block:: python

                        'data_aug_op_config': {
                            'kwargs': {
                                'lang': 'en',
                                'use_gpu': False
                            }
                        }

            - augment_pack_names:
                A dict specifies the DataPacks to augment and their output
                names. It should be key-value pairs where the key is the
                input DataPack name, and the value is the output DataPack
                name after augmentation.

                If empty, all the DataPacks will be augmented, and the output
                names will be automatically generated by prepending
                an `'augmented_'` prefix.

                Example:
                    .. code-block:: python

                        'data_aug_op_config': {
                            'kwargs': {
                                'src': 'aug_src',
                                'tgt': 'aug_tgt'
                            }
                        }
        """
        config = super().default_configs()
        config.update({
            'augment_entry': 'ft.onto.base_ontology.Sentence',
            'other_entry_policy': {
                'type': '',
                'kwargs': {}
            },
            'type': 'data_augmentation_op',
            'data_aug_op': '',
            'data_aug_op_config': {
                'type': '',
                'kwargs': {}
            },
            'augment_pack_names': {
                'type': '',
                'kwargs': {}
            }
        })
        return config
