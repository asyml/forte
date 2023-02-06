# Copyright 2022 The Forte Authors. All Rights Reserved.
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
SkeletonOp is the most low level Op used for Data Augmentation.
Through this Op, users can use in-built utility functions to implement
their own augmentation logic.
"""
from collections import defaultdict
from copy import deepcopy
from typing import (
    DefaultDict,
    Iterable,
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Set,
    cast,
)
from abc import abstractmethod
from bisect import bisect_left, bisect_right
from sortedcontainers.sorteddict import SortedDict
from sortedcontainers.sortedlist import SortedList
from forte.data.data_pack import DataPack
from forte.common.configuration import Config
from forte.data.span import Span
from forte.data.ontology.top import (
    Link,
    Group,
    Annotation,
)
from forte.data.ontology.core import Entry
from forte.common.configurable import Configurable
from forte.utils.utils import create_class_with_kwargs, get_class


__all__ = ["BaseDataAugmentationOp"]


class BaseDataAugmentationOp(Configurable):
    r"""
    The SkeletonOp is the most basic augmentation Op that
    gives users the most amount of freedom to implement their
    logic of augmentation. The users are expected to use the
    provided utility functions to implement their own augmentation
    logic which will then be substantiated into new data packs. This
    Op requires the users to have a relatively stronger understanding of
    Forte's internal setup.
    """

    def __init__(self, configs: Union[Config, Dict[str, Any]]) -> None:

        # :attr:`_replaced_annos`: {datapack id: SortedList[span, new text]}
        # It records the spans replaced by new texts.
        # It is a map from datapack id to a list of tuples
        # (span, new text) inserted by :func:`replace`.
        # The records stored in `_replaced_annos` will then be used to
        # create a new data pack with updated text.

        self._replaced_annos: DefaultDict[
            int, SortedList[Tuple[Span, str]]
        ] = defaultdict(lambda: SortedList([], key=lambda x: x[0]))

        # :attr:`_inserted_text`: {datapack id: Dict{position: length}}
        # It records the position and length of inserted spans,
        # mapping from datapack id to a dictionary (position -> length)
        # inserted by :func:`insert`.
        # The position is the index in the original datapack
        # of insertion, and the length is the length of the inserted string.
        self._inserted_text: DefaultDict[
            int, SortedDict[int, Tuple[int, str]]
        ] = defaultdict(lambda: SortedDict({}))

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

        self.configs = self.make_configs(configs)

    def perform_augmentation(self, input_pack: DataPack) -> DataPack:
        r"""
        Function to apply the defined augmentation function and
        instantiating it into a new data pack. This data pack is
        then returned.

        Args:
            data_pack: The Datapack holding the replaced annotations.
        Returns:
            A new data pack holds the text after replacement.
        """
        self.augment(input_pack)
        augmented_data_pack = self._apply_augmentations(input_pack)
        return augmented_data_pack

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
            A bool value indicating whether the copy happens successfully.
        """

        # If the entry has been copied, return True.
        if entry.tid in entry_map:
            return True

        # The entry should be either Link or Group.
        is_link: bool = isinstance(entry, Link)

        # Get the children entries.
        children: List[Entry]
        if is_link:
            children = [entry.get_parent(), entry.get_child()]  # type: ignore
        else:
            children = entry.get_members()  # type: ignore

        # Copy the children entries.
        new_children: List[Entry] = []
        for child_entry in children:
            if isinstance(child_entry, (Link, Group)):
                # Recursively copy the children Links/Groups.
                if not self._copy_link_or_group(
                    child_entry, entry_map, new_pack
                ):
                    return False
            else:
                # Children Annotation must have been copied.
                if child_entry.tid not in entry_map:
                    return False
            new_child: Entry = new_pack.get_entry(entry_map[child_entry.tid])
            new_children.append(new_child)

        # Create the new entry and add to the new pack.
        new_entry: Entry
        if is_link:
            entry = cast(Link, entry)
            new_link_parent: Entry = new_children[0]
            new_link_child: Entry = new_children[1]
            new_entry = type(entry)(
                new_pack, new_link_parent, new_link_child  # type: ignore
            )
        else:
            entry = cast(Group, entry)
            new_entry = type(entry)(new_pack, new_children)  # type: ignore
        new_pack.add_entry(new_entry)
        entry_map[entry.tid] = new_entry.tid
        return True

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
        if len(self._replaced_annos[pid]) == 0:
            return False
        ind: int = (
            bisect_left(self._replaced_annos[pid], (Span(begin, begin), "")) - 1
        )
        if ind < 0:
            ind += 1

        while ind < len(self._replaced_annos[pid]):
            span: Span = self._replaced_annos[pid][ind][0]
            if not (span.begin >= end or span.end <= begin):
                return True
            if span.begin > end:
                break
            ind += 1

        return False

    def modify_index(
        self,
        index: int,
        # Both of the following spans should be SortedList.
        # Use List to avoid typing errors.
        old_spans: List[Span],
        new_spans: List[Span],
        is_begin: bool,
        is_inclusive: bool,
    ) -> int:

        r"""
        A helper function to map an index before replacement
        to the index after replacement.
        An index is the character offset in the data pack.
        The `old_spans` are the inputs of replacement, and the new_spans
        are the outputs. Each of the span has start and end index.
        The `old_spans` and `new_spans` are anchors for the mapping,
        because we depend on them to determine the position change of the
        index.
        Given an index, the function will find its the nearest
        among the old spans before the index, and calculate the difference
        between the position of the old span and its corresponding new span.
        The position change is then applied to the input index. An updated
        index is then calculated and returned.
        An inserted span might be included as a part of another span.
        For example, given a sentence "I love NLP.", if we insert a
        token "Yeah" at the beginning of the sentence(index=0), the Sentence
        should include the new Token, i.e., the Sentence will have a
        start index equals to 0. In this case, the parameter is_inclusive
        should be True. However, for another Token "I", it should not include
        the new token, so its start index will be larger than 0.
        The parameter in_inclusive should be False.
        The input index could be the start or end index of a span, i.e., the
        left or right boundary of the span. If there is an insertion in the span,
        we should treat the two boundaries in different ways. For example,
        we have a paragraph with two sentences "I love NLP! You love NLP too."
        If we append another "!" to the end of the first sentence, when modifying
        the end index of the first Sentence, it should be pushed right to include
        the extra exclamation. In this case, the is_begin is False. However, if
        we prepend an "And" to the second sentence, when modifying the start index
        of the second Sentence, it should be pushed left to include the new Token.
        In this case, the `is_begin` is True.

        Args:
            index: The index to map.
            old_spans: The spans before replacement. It should be
                a sorted list in ascending order.
            new_spans: The spans after replacement. It should be
                a sorted list in ascending order.
            is_begin: True if the input index is the start index of a span.
            is_inclusive: True if the span constructed by the aligned
                index should include inserted spans.

        Returns:
            The aligned index.

        If the old spans are [0, 1], [2, 3], [4, 6],
        the new spans are [0, 4], [5, 7], [8, 11],
        the input index is 3, and there are no insertions,
        the algorithm will first locate the last span with
        a begin index less or equal than the target index,
        ([2,3]), and find the corresponding span in new spans([5,7]).
        Then we calculate the delta index(7-3=4) and update our
        input index(3+4=7). The output then is 7.

        Note that when the input index locates inside the old spans,
        instead of on the boundary of the spans, we compute the return
        index so that it maintains the same offset to the begin of the
        span it belongs to. In the above example, if we change the input
        index from 3 to 5, the output will become 9, because we locates
        the input index in the third span [4, 6] and use the same offset
        5-4=1 to calculate the output 8+1=9.

        When insertion is considered, there will be spans
        with the same begin index, for example,
        [0, 1], [1, 1], [1, 2]. The span [1, 1] indicates an insertion
        at index 1, because the insertion can be considered as a
        replacement of an empty input span, with a length of 0.
        The output will be affected by whether to include the inserted
        span(is_inclusive), and whether the input index is a begin or
        end index of its span(is_begin).

        If the old spans are [0, 1], [1, 1], [1, 2],
        the new spans are [0, 2], [2, 4], [4, 5],
        the input index is 1, the output will be 2 if both
        is_inclusive and is_begin are True,
        because the inserted [1, 1] should be included in the span.
        If the `is_inclusive=True` but `is_begin=False`, the output will be
        4 because the index is an end index of the span.
        """

        # Get the max index for binary search.
        max_index: int = old_spans[-1].end + 1
        max_index = max(max_index, index)

        # This is the last span that has a start index less than
        # the input index. The position change of this span determines
        # the modification we will apply to the input index.
        last_span_ind: int = bisect_right(old_spans, Span(index, max_index)) - 1

        # If there is an inserted span, it will always be the first of
        # those spans with the same begin index. For example, given spans
        # [1, 1], [1, 2], The inserted span [1, 1] will be in the front of
        # replaced span [1, 2], because it has the smallest end index.
        if last_span_ind >= 0:
            if is_inclusive:
                if is_begin:
                    # When inclusive, move the begin index
                    # to the left to include the inserted span.
                    if (
                        last_span_ind > 0
                        and old_spans[last_span_ind - 1].begin == index
                    ):
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
                    if (
                        last_span_ind > 0
                        and old_spans[last_span_ind - 1].begin == index
                    ):
                        # Old spans: [0, 1], [1, 1], [1, 3]
                        # Target index: 1
                        # Change last_span_index from 2 to 0
                        # to exclude the [1, 1] span.
                        last_span_ind -= 2
                    elif (
                        old_spans[last_span_ind].begin == index
                        and old_spans[last_span_ind].end == index
                    ):
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
        delta_index: int = (
            new_spans[last_span_ind].begin - old_spans[last_span_ind].begin
        )

        if (
            old_spans[last_span_ind].begin == old_spans[last_span_ind].end
            and old_spans[last_span_ind].begin == index
            and is_begin
            and is_inclusive
        ):
            return index + delta_index

        if new_spans[last_span_ind].begin == new_spans[last_span_ind].end and (
            old_spans[last_span_ind].begin
            <= index
            <= old_spans[last_span_ind].end
        ):
            return new_spans[last_span_ind].begin

        if old_spans[last_span_ind].end <= index:
            # Use the span's end index as anchor, if possible.
            delta_index = (
                new_spans[last_span_ind].end - old_spans[last_span_ind].end
            )
        return index + delta_index

    def _apply_augmentations(
        self,
        data_pack: DataPack,
    ) -> DataPack:

        r"""
        The objective of this function is to actualize the augmentations
        proposed by the augment function. It will copy and update the text
        of datapack and auto-align the annotation spans. The links are also
        copied if its parent & child are both present in the new pack.
        The groups are copied if all its members are present
        in the new pack.

        Args:
            data_pack: The Datapack holding the replaced annotations.

        Returns:
            A new data_pack holds the text after replacement. The annotations
            in the original data pack will be copied and auto-aligned as
            instructed by the "other_entry_policy" in the configuration.
            The links and groups will be copied if their members are copied.
            New annotations added by the `insert_annotated_spans` function
            will also be added to the newly created data pack. Conversely, if
            annotation is deleted by the `delete_annotation` function or an annotation
            exists within a span that is deleted by the `delete_span` function, it will
            not be added to the new data pack.
        """

        replaced_annotations = self._replaced_annos[data_pack.pack_id]

        if len(replaced_annotations) == 0:
            return deepcopy(data_pack)

        spans: List[Span] = [span for span, _ in replaced_annotations]
        replacement_strs: List[str] = [
            replacement_str for _, replacement_str in replaced_annotations
        ]

        # Get the new text for the new data pack.
        new_text: str = ""
        for i, span in enumerate(spans):
            new_span_str = replacement_strs[i]
            # First, get the gap text between last and this span.
            last_span_end: int = spans[i - 1].end if i > 0 else 0
            gap_text: str = data_pack.text[last_span_end : span.begin]
            new_text += gap_text
            # Then, append the replaced new text.
            new_text += new_span_str

        # Finally, append to new_text the text after the last span.
        new_text += data_pack.text[spans[-1].end :]

        # Get the span (begin, end) before and after replacement.
        new_spans: List[Span] = []

        # Bias is the delta between the beginning
        # indices before & after replacement.
        bias: int = 0
        for i, span in enumerate(spans):
            old_begin: int = span.begin
            old_end: int = span.end
            new_begin: int = old_begin + bias
            new_end = new_begin + len(replacement_strs[i])
            new_spans.append(Span(new_begin, new_end))
            bias = new_end - old_end

        new_pack: DataPack = DataPack()
        new_pack.set_text(new_text)

        entry_map: Dict[int, int] = {}
        insert_ind: int = 0
        pid: int = data_pack.pack_id

        # Only iterate over those entries that are necessary. ie. the
        # ones that are inserted or are present in the other_entry_policy
        # config.
        existing_entries = self.configs["other_entry_policy"].keys()

        new_entries: Dict[str, List[Tuple[int, int]]] = {}
        for pos, data in self._inserted_text[pid].items():
            new_entries[data[1]] = new_entries.get(data[1], []) + [
                (pos, data[0])
            ]

        entries_to_copy: Set[str] = set(
            list(existing_entries)
            + [val for val in new_entries if val is not None]
        )

        def _insert_new_span(
            entry_class: str,
            insert_ind: int,
            inserted_annos: List[Tuple[int, int]],
            new_pack: DataPack,
            spans: List[Span],
            new_spans: List[Span],
        ):
            """
            An internal helper function for insertion.

            Args:
                entry_class: The new annotation type to be created.
                insert_ind: The index to be insert.
                inserted_annos: The annotation span information to be inserted.
                new_pack: The new data pack to insert the annotation.
                spans: The original spans before replacement, should be
                  a sorted ascending list.
                new_spans: The original spans before replacement, should be
                  a sorted ascending list.
            """
            pos: int
            length: int
            pos, length = inserted_annos[insert_ind]
            if entry_class is None:
                return
            insert_end: int = self.modify_index(
                pos,
                spans,
                new_spans,
                is_begin=False,
                # Include the inserted span itself.
                is_inclusive=True,
            )
            insert_begin: int = insert_end - length
            new_anno = create_class_with_kwargs(
                entry_class,
                {"pack": new_pack, "begin": insert_begin, "end": insert_end},
            )
            new_pack.add_entry(new_anno)

        # Iterate over all the original entries and modify their spans.
        for entry_to_copy in entries_to_copy:

            class_to_copy = get_class(entry_to_copy)
            insert_ind = 0

            if not issubclass(class_to_copy, Annotation):
                raise AttributeError(
                    f"The entry type to copy from [{entry_to_copy}] is not "
                    f"a sub-class of 'forte.data.ontology.top.Annotation'."
                )

            if entry_to_copy not in new_entries:
                new_entries[entry_to_copy] = []

            orig_annos: Iterable[Annotation] = data_pack.get(class_to_copy)

            for orig_anno in orig_annos:
                # Dealing with insertion/deletion only for augment_entry.
                while (
                    insert_ind < len(new_entries[entry_to_copy])
                    and new_entries[entry_to_copy][insert_ind][0]
                    <= orig_anno.begin
                ):
                    # Preserve the order of the spans with merging sort.
                    # It is a 2-way merging from the inserted spans
                    # and original spans based on the begin index.
                    _insert_new_span(
                        entry_to_copy,
                        insert_ind,
                        new_entries[entry_to_copy],
                        new_pack,
                        spans,
                        new_spans,
                    )
                    insert_ind += 1

                # Deletion
                if orig_anno.tid in self._deleted_annos_id[pid]:
                    continue

                # Auto align the spans.
                span_new_begin: int = orig_anno.begin
                span_new_end: int = orig_anno.end

                if (entry_to_copy in new_entries) or self.configs[
                    "other_entry_policy"
                ][entry_to_copy] == "auto_align":
                    # Only inclusive when the entry is not augmented.
                    # E.g.: A Sentence include the inserted Token on the edge.
                    # E.g.: A Token shouldn't include a nearby inserted Token.
                    is_inclusive = (entry_to_copy in existing_entries) and (
                        entry_to_copy not in new_entries
                    )
                    span_new_begin = self.modify_index(
                        orig_anno.begin, spans, new_spans, True, is_inclusive
                    )
                    span_new_end = self.modify_index(
                        orig_anno.end, spans, new_spans, False, is_inclusive
                    )

                # If an annotation is within a deleted span, there
                # is no need to add that annotation to the new data pack
                if span_new_begin == span_new_end:
                    continue

                new_anno = create_class_with_kwargs(
                    entry_to_copy,
                    {
                        "pack": new_pack,
                        "begin": span_new_begin,
                        "end": span_new_end,
                    },
                )
                new_pack.add_entry(new_anno)
                entry_map[orig_anno.tid] = new_anno.tid

            # Deal with spans after the last annotation in the original pack.
            while insert_ind < len(new_entries[entry_to_copy]):
                _insert_new_span(
                    entry_to_copy,
                    insert_ind,
                    new_entries[entry_to_copy],
                    new_pack,
                    spans,
                    new_spans,
                )
                insert_ind += 1

        # Iterate over and copy the links/groups in the datapack.
        for link in data_pack.get(Link):
            self._copy_link_or_group(link, entry_map, new_pack)
        for group in data_pack.get(Group):
            self._copy_link_or_group(group, entry_map, new_pack)

        self._data_pack_map[pid] = new_pack.pack_id
        self._entry_maps[pid] = entry_map
        return new_pack

    def insert_span(
        self, inserted_text: str, data_pack: DataPack, pos: int
    ) -> bool:
        r"""
        This is a utility function to insert a new text span to a data pack.
        The inserted span will not have any annotation associated with it.
        After getting the inserted text, it will register the input & output
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

        pid: int = data_pack.pack_id
        if self._overlap_with_existing(pid, pos, pos):
            return False

        if pos not in self._inserted_text[pid]:
            self._replaced_annos[pid].add((Span(pos, pos), inserted_text))
            self._inserted_text[pid][pos] = (len(inserted_text), None)
            return True
        return False

    def insert_annotated_span(
        self,
        inserted_text: str,
        data_pack: DataPack,
        pos: int,
        annotation_type: str,
    ) -> bool:
        r"""
        This is a utility function to insert a new annotation to a data pack.
        After getting the inserted text, it will register the input & output
        for later batch process of building the new data pack.
        The insertion at each position can only occur once. If there
        is already an insertion at current position, it will abort the
        insertion and return False.

        Args:
            inserted_text: The text string to insert.
            data_pack: The datapack for insertion.
            pos: The position(index) of insertion.
            annotation_type: The type of annotation this span represents.
        Returns:
            A bool value. True if the insertion happened, False otherwise.

        """
        pid: int = data_pack.pack_id
        if self._overlap_with_existing(pid, pos, pos):
            return False

        if pos not in self._inserted_text[pid]:
            self._replaced_annos[pid].add((Span(pos, pos), inserted_text))
            self._inserted_text[pid][pos] = (
                len(inserted_text),
                annotation_type,
            )
            return True
        return False

    def delete_annotation(self, input_anno: Annotation) -> bool:
        r"""
        This is a utility function to delete an annotation. If the same
        annotation is tried to be deleted twice, the function will terminate
        and return False.

        Args:
            input_anno: The annotation to remove.
        Returns:
            A bool value. True if the deletion happened, False otherwise.
        """
        pid: int = input_anno.pack.pack_id
        if self._overlap_with_existing(pid, input_anno.begin, input_anno.end):
            return False

        self._replaced_annos[pid].add((input_anno.span, ""))
        self._deleted_annos_id[pid].add(input_anno.tid)
        return True

    def delete_span(self, data_pack: DataPack, begin: int, end: int) -> bool:
        r"""
        This is a utility function to delete a span of text. If the same
        annotation is tried to be deleted twice, the function will terminate
        and return False. If the method deletes only a portion of an existing annotation,
        The annotation will be calibrated to represent the remaining part of the span. Moreover,
        if the deleted span covers an entire annotation, the entire annotation will be deleted.

        Args:
            input_anno: The annotation to remove.
            begin: The starting position of the span to delete
            end: The ending position of the span to delete
        Returns:
            A bool value. True if the deletion happened, False otherwise.
        """

        pid: int = data_pack.pack_id
        if self._overlap_with_existing(pid, begin, end):
            return False

        self._replaced_annos[pid].add((Span(begin, end), ""))
        return True

    def replace_annotations(
        self, replacement_anno: Annotation, replaced_text
    ) -> bool:
        r"""
        This is a utility function to record specifically a replacement
        of the text in an annotation. With this function, the text inside
        annotation can be replaced with another text. If the same annotation
        is tried to be replaced twice, the function will terminate and
        return False.

        Args:
            input_anno: The annotation to replace.
        Returns:
            A bool value. True if the replacement happened, False otherwise.
        """
        pid: int = replacement_anno.pack.pack_id
        if self._overlap_with_existing(
            pid, replacement_anno.begin, replacement_anno.end
        ):
            return False

        self._replaced_annos[pid].add(
            (Span(replacement_anno.begin, replacement_anno.end), replaced_text)
        )
        return True

    def clear_states(self):
        r"""
        This function clears the states. It should be
        called after processing a multipack.
        """
        self._replaced_annos.clear()
        self._inserted_text.clear()
        self._deleted_annos_id.clear()
        self._data_pack_map.clear()
        self._entry_maps.clear()

    def get_maps(self) -> Tuple[Dict[int, int], Dict[int, Dict[int, int]]]:
        r"""
        This function simply returns the produced data pack
        and entry maps after augmentation.

        Returns:
            A tuple of two elements. The first element is the data pack
            map (dict) and the second element is the entry maps (dict)
        """
        return self._data_pack_map, self._entry_maps

    @abstractmethod
    def augment(self, data_pack: DataPack) -> bool:
        r"""
        This method is left to be implemented by the user of this Op.
        The user can use any of the given utility functions to perform
        augmentation.

        Args:
            data_pack: the input data pack to augment

        Returns:
            A boolean value indicating if the augmentation
            was successful (True) or unsuccessful (False).
        """
        raise NotImplementedError

    @classmethod
    def default_configs(cls):
        """
        Returns:
            A dictionary with the default config for this processor.
        Following are the keys for this dictionary:
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
                            "ft.onto.base_ontology.Document": "auto_align",
                            "ft.onto.base_ontology.Sentence": "auto_align",
                        }
        """
        return {
            "other_entry_policy": {
                "ft.onto.base_ontology.Sentence": "auto_align",
                "ft.onto.base_ontology.Token": "auto_align",
            },
            "@no_typecheck": ["other_entry_policy"],
        }
