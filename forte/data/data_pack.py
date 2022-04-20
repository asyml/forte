# Copyright 2019 The Forte Authors. All Rights Reserved.
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

import logging
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Type,
    Union,
    Any,
    Set,
    Callable,
    Tuple,
)

import numpy as np
from sortedcontainers import SortedList

from forte.common.exception import (
    ProcessExecutionException,
    UnknownOntologyClassException,
)
from forte.data import data_utils_io
from forte.data.base_pack import BaseMeta, BasePack
from forte.data.index import BaseIndex
from forte.data.ontology.core import Entry
from forte.data.ontology.core import EntryType
from forte.data.ontology.top import (
    Annotation,
    Link,
    Group,
    SinglePackEntries,
    Generics,
    AudioAnnotation,
)
from forte.data.span import Span
from forte.data.types import ReplaceOperationsType, DataRequest
from forte.utils import get_class

logger = logging.getLogger(__name__)

__all__ = ["Meta", "DataPack", "DataIndex"]


class Meta(BaseMeta):
    r"""Basic Meta information associated with each instance of
    :class:`~forte.data.data_pack.DataPack`.

    Args:
        pack_name:  An name to identify the data pack, which is helpful in
           situation like serialization. It is suggested that the packs should
           have different doc ids.
        language: The language used by this data pack, default is English.
        span_unit: The unit used for interpreting the Span object of this
          data pack. Default is character.
        sample_rate: An integer specifying the sample rate of audio payload.
          Default is None.
        info: Store additional string based information that the user add.
    Attributes:
        pack_name:  storing the provided `pack_name`.
        language: storing the provided `language`.
        sample_rate: storing the provided `sample_rate`.
        info: storing the provided `info`.
        record: Initialized as a dictionary. This is not a required field.
            The key of the record should be the entry type and values should
            be attributes of the entry type. All the information would be used
            for consistency checking purpose if the pipeline is initialized with
            `enforce_consistency=True`.
    """

    def __init__(
        self,
        pack_name: Optional[str] = None,
        language: str = "eng",
        span_unit: str = "character",
        sample_rate: Optional[int] = None,
        info: Optional[Dict[str, str]] = None,
    ):
        super().__init__(pack_name)
        self.language = language
        self.span_unit = span_unit
        self.sample_rate: Optional[int] = sample_rate
        self.record: Dict[str, Set[str]] = {}
        self.info: Dict[str, str]
        if info is None:
            self.info = {}
        else:
            self.info = info


def as_entry_type(entry_type: Union[str, Type[EntryType]]):
    entry_type_: Type[EntryType]
    if isinstance(entry_type, str):
        entry_type_ = get_class(entry_type)
        if not issubclass(entry_type_, Entry):
            raise ValueError(
                f"The specified entry type [{entry_type}] "
                f"does not correspond to a "
                f"`forte.data.ontology.core.Entry` class"
            )
    else:
        entry_type_ = entry_type
    return entry_type_


def as_sorted_error_check(entries: List[EntryType]) -> SortedList:
    """
    Given a list of entries, return a sorted list of it. If unknown entry
    classes are seen during this process,
    a :class:`~forte.common.exception.UnknownOntologyClassException` exception will be
    thrown.

    Args:
        entries: A list of entries to be converted.

    Returns: Sorted list of the input entries.
    """
    try:
        return SortedList(entries)
    except TypeError as e:
        for entry in entries:
            if isinstance(entry, Dict) and "py/object" in entry:
                entry_class = entry["py/object"]
                try:
                    get_class(entry_class)
                except ValueError:
                    raise UnknownOntologyClassException(
                        f"Cannot deserialize ontology type {entry_class}, "
                        f"make sure it is included in the PYTHONPATH."
                    ) from e


class DataPack(BasePack[Entry, Link, Group]):
    # pylint: disable=too-many-public-methods, unused-private-member
    r"""A :class:`~forte.data.data_pack.DataPack` contains a piece of natural language text and a
    collection of NLP entries (annotations, links, and groups). The natural
    language text could be a document, paragraph or in any other granularity.

    Args:
        pack_name: A name for this data pack.
    """


class DataIndex(BaseIndex):
    r"""A set of indexes used in :class:`~forte.data.data_pack.DataPack`, note that this class is
    used by the `DataPack` internally.

    #. :attr:`entry_index`, the index from each tid to the corresponding entry
    #. :attr:`type_index`, the index from each type to the entries of
       that type
    #. :attr:`component_index`, the index from each component to the
       entries generated by that component
    #. :attr:`link_index`, the index from child
       (:attr:`link_index["child_index"]`)and parent
       (:attr:`link_index["parent_index"]`) nodes to links
    #. :attr:`group_index`, the index from group members to groups.
    #. :attr:`_coverage_index`, the index that maps from an annotation to
       the entries it covers. :attr:`_coverage_index` is a dict of dict, where
       the key is a tuple of the outer entry type and the inner entry type.
       The outer entry type should be an annotation type. The value is a dict,
       where the key is the tid of the outer entry, and the value is a set of
       ``tid`` that are covered by the outer entry. We say an Annotation A covers
       an entry E if one of the following condition is met:
       1. E is of Annotation type, and that E.begin >= A.begin, E.end <= E.end
       2. E is of Link type, and both E's parent and child node are Annotation
       that are covered by A.

    """

    def __init__(self):
        super().__init__()
        self._coverage_index: Dict[
            Tuple[Type[Union[Annotation, AudioAnnotation]], Type[EntryType]],
            Dict[int, Set[int]],
        ] = {}
        self._coverage_index_valid = True

    def remove_entry(self, entry: EntryType):
        super().remove_entry(entry)
        self.deactivate_coverage_index()

    @property
    def coverage_index_is_valid(self):
        return self._coverage_index_valid

    def activate_coverage_index(self):
        self._coverage_index_valid = True

    def deactivate_coverage_index(self):
        self._coverage_index_valid = False

    def coverage_index(
        self,
        outer_type: Type[Union[Annotation, AudioAnnotation]],
        inner_type: Type[EntryType],
    ) -> Optional[Dict[int, Set[int]]]:
        r"""Get the coverage index from ``outer_type`` to ``inner_type``.

        Args:
            outer_type: an annotation or `AudioAnnotation` type.
            inner_type: an entry type.

        Returns:
            If the coverage index does not exist, return `None`. Otherwise,
            return a dict.
        """
        if not self.coverage_index_is_valid:
            return None
        return self._coverage_index.get((outer_type, inner_type))

    def get_covered(
        self,
        data_pack: DataPack,
        context_annotation: Union[Annotation, AudioAnnotation],
        inner_type: Type[EntryType],
    ) -> Set[int]:
        """
        Get the entries covered by a certain context annotation

        Args:
            data_pack: The data pack to search for.
            context_annotation: The context annotation to search in.
            inner_type: The inner type to be searched for.

        Returns:
            Entry ID of type `inner_type` that is covered by
            `context_annotation`.
        """
        context_type = context_annotation.__class__
        if self.coverage_index(context_type, inner_type) is None:
            self.build_coverage_index(data_pack, context_type, inner_type)
        assert self._coverage_index is not None
        return self._coverage_index.get((context_type, inner_type), {}).get(
            context_annotation.tid, set()
        )

    def build_coverage_index(
        self,
        data_pack: DataPack,
        outer_type: Type[Union[Annotation, AudioAnnotation]],
        inner_type: Type[EntryType],
    ):
        r"""Build the coverage index from ``outer_type`` to ``inner_type``.

        Args:
            data_pack: The data pack to build coverage for.
            outer_type: an annotation or `AudioAnnotation` type.
            inner_type: an entry type, can be Annotation, Link, Group,
                `AudioAnnotation`.
        """
        if not issubclass(
            inner_type, (Annotation, Link, Group, AudioAnnotation)
        ):
            raise ValueError(f"Do not support coverage index for {inner_type}.")

        if not self.coverage_index_is_valid:
            self._coverage_index = {}

        # prevent the index from being used during construction
        self.deactivate_coverage_index()

        # TODO: tests and documentations for the edge cases are missing. i.e. we
        #  are not clear about what would happen if the covered annotation
        #  is the same as the covering annotation, or if their spans are the
        #  same.
        self._coverage_index[(outer_type, inner_type)] = {}
        for range_annotation in data_pack.get_entries_of(outer_type):
            if isinstance(range_annotation, (Annotation, AudioAnnotation)):
                entries = data_pack.get(inner_type, range_annotation)
                entry_ids = {e.tid for e in entries}
                self._coverage_index[(outer_type, inner_type)][
                    range_annotation.tid
                ] = entry_ids

        self.activate_coverage_index()

    def have_overlap(
        self,
        entry1: Union[Annotation, int, AudioAnnotation],
        entry2: Union[Annotation, int, AudioAnnotation],
    ) -> bool:
        r"""Check whether the two annotations have overlap in span.

        Args:
            entry1: An
                :class:`Annotation` or :class:`AudioAnnotation` object to be
                checked, or the tid of the Annotation.
            entry2: Another
                :class:`Annotation` or :class:`AudioAnnotation` object to be
                checked, or the tid of the Annotation.
        """
        entry1_: Union[Annotation, AudioAnnotation] = (
            self._entry_index[entry1]
            if isinstance(entry1, (int, np.integer))
            else entry1
        )
        entry2_: Union[Annotation, AudioAnnotation] = (
            self._entry_index[entry2]
            if isinstance(entry2, (int, np.integer))
            else entry2
        )

        if not isinstance(entry1_, (Annotation, AudioAnnotation)):
            raise TypeError(
                f"'entry1' should be an instance of Annotation or `AudioAnnotation`,"
                f" but get {type(entry1)}"
            )

        if not isinstance(entry2_, (Annotation, AudioAnnotation)):
            raise TypeError(
                f"'entry2' should be an instance of Annotation or `AudioAnnotation`,"
                f" but get {type(entry2)}"
            )

        if (
            isinstance(entry1_, Annotation)
            and isinstance(entry2_, AudioAnnotation)
        ) or (
            isinstance(entry1_, AudioAnnotation)
            and isinstance(entry2_, Annotation)
        ):
            raise TypeError(
                "'entry1' and 'entry2' should be the same type of entry, "
                f"but get type(entry1)={type(entry1_)}, "
                f"typr(entry2)={type(entry2_)}"
            )

        return not (
            entry1_.begin >= entry2_.end or entry1_.end <= entry2_.begin
        )

    def in_span(self, inner_entry: Union[int, Entry], span: Span) -> bool:
        r"""Check whether the ``inner entry`` is within the given ``span``. The
        criterion are as followed:

        Annotation entries: they are considered in a span if the
        begin is not smaller than `span.begin` and the end is not larger than
        `span.end`.

        Link entries: if the parent and child of the links are both
        `Annotation` type, this link will be considered in span if both parent
        and child are :meth:`~forte.data.data_pack.DataIndex.in_span` of the
        provided `span`. If either the parent and
        the child is not of type `Annotation`, this function will always return
        `False`.

        Group entries: if the child type of the group is `Annotation` type,
        then the group will be considered in span if all the elements are
        :meth:`~forte.data.data_pack.DataIndex.in_span` of the provided `span`.
        If the child type is not `Annotation`
        type, this function will always return `False`.

        Other entries (i.e Generics and `AudioAnnotation`): they will not be
        considered :meth:`~forte.data.data_pack.DataIndex.in_span` of any
        spans. The function will always return
        `False`.

        Args:
            inner_entry: The inner entry object to be checked
             whether it is within ``span``. The argument can be the entry id
             or the entry object itself.
            span: A :class:`~forte.data.span.Span` object to be checked. We will check
                whether the ``inner_entry`` is within this span.

        Returns:
            True if the `inner_entry` is considered to be in span of the
            provided span.
        """
        # The reason of this check is that the get_data method will use numpy
        # integers. This might create problems when other unexpected integers
        # are used.
        if isinstance(inner_entry, (int, np.integer)):
            inner_entry = self._entry_index[inner_entry]

        inner_begin = -1
        inner_end = -1

        if isinstance(inner_entry, Annotation):
            inner_begin = inner_entry.begin
            inner_end = inner_entry.end
        elif isinstance(inner_entry, Link):
            if not issubclass(inner_entry.ParentType, Annotation):
                return False

            if not issubclass(inner_entry.ChildType, Annotation):
                return False

            child = inner_entry.get_child()
            parent = inner_entry.get_parent()

            if not isinstance(child, Annotation) or not isinstance(
                parent, Annotation
            ):
                # Cannot check in_span for non-annotations.
                return False

            child_: Annotation = child
            parent_: Annotation = parent

            inner_begin = min(child_.begin, parent_.begin)
            inner_end = max(child_.end, parent_.end)
        elif isinstance(inner_entry, Group):
            if not issubclass(inner_entry.MemberType, Annotation):
                return False

            for mem in inner_entry.get_members():
                mem_: Annotation = mem  # type: ignore
                if inner_begin == -1:
                    inner_begin = mem_.begin
                inner_begin = min(inner_begin, mem_.begin)
                inner_end = max(inner_end, mem_.end)
        else:
            # Generics, AudioAnnotation, or other user defined types will not
            # be check here.
            return False
        return inner_begin >= span.begin and inner_end <= span.end

    def in_audio_span(self, inner_entry: Union[int, Entry], span: Span) -> bool:
        r"""Check whether the ``inner entry`` is within the given audio span.
        This method is identical to
        :meth::meth:`~forte.data.data_pack.DataIndex.in_span` except that it
        operates on
        the audio payload of datapack. The criterion are as followed:

        `AudioAnnotation` entries: they are considered in a span if the
        begin is not smaller than `span.begin` and the end is not larger than
        `span.end`.

        Link entries: if the parent and child of the links are both
        `AudioAnnotation` type, this link will be considered in span if both
        parent and child are :meth:`~forte.data.data_pack.DataIndex.in_span` of
        the provided `span`. If either the
        parent and the child is not of type `AudioAnnotation`, this function
        will always return `False`.

        Group entries: if the child type of the group is `AudioAnnotation`
        type,
        then the group will be considered in span if all the elements are
        :meth:`~forte.data.data_pack.DataIndex.in_span` of the provided `span`.
        If the child type is not
        `AudioAnnotation` type, this function will always return `False`.

        Other entries (i.e Generics and Annotation): they will not be
        considered
        :meth:`~forte.data.data_pack.DataIndex.in_span` of any spans. The
        function will always return `False`.

        Args:
            inner_entry: The inner entry object to be checked
                whether it is within ``span``. The argument can be the entry id
                or the entry object itself.
            span: A :class:`~forte.data.span.Span` object to be checked.
                We will check whether the ``inner_entry`` is within this span.

        Returns:
            True if the `inner_entry` is considered to be in span of the
            provided span.
        """
        # The reason of this check is that the get_data method will use numpy
        # integers. This might create problems when other unexpected integers
        # are used.
        if isinstance(inner_entry, (int, np.integer)):
            inner_entry = self._entry_index[inner_entry]

        inner_begin = -1
        inner_end = -1

        if isinstance(inner_entry, AudioAnnotation):
            inner_begin = inner_entry.begin
            inner_end = inner_entry.end
        elif isinstance(inner_entry, Link):
            if not (
                issubclass(inner_entry.ParentType, AudioAnnotation)
                and issubclass(inner_entry.ChildType, AudioAnnotation)
            ):
                return False

            child = inner_entry.get_child()
            parent = inner_entry.get_parent()

            if not isinstance(child, AudioAnnotation) or not isinstance(
                parent, AudioAnnotation
            ):
                # Cannot check in_span for non-AudioAnnotation.
                return False

            child_: AudioAnnotation = child
            parent_: AudioAnnotation = parent

            inner_begin = min(child_.begin, parent_.begin)
            inner_end = max(child_.end, parent_.end)
        elif isinstance(inner_entry, Group):
            if not issubclass(inner_entry.MemberType, AudioAnnotation):
                return False

            for mem in inner_entry.get_members():
                mem_: AudioAnnotation = mem  # type: ignore
                if inner_begin == -1:
                    inner_begin = mem_.begin
                inner_begin = min(inner_begin, mem_.begin)
                inner_end = max(inner_end, mem_.end)
        else:
            # Generics, Annotation, or other user defined types will not be
            # check here.
            return False
        return inner_begin >= span.begin and inner_end <= span.end
