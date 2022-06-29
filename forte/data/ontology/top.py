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
from dataclasses import dataclass
from functools import total_ordering
from lib2to3.pgen2.token import OP
from typing import Optional, Tuple, Type, Any, Dict, Union, Iterable, List

import logging
import numpy as np
from forte.data.base_pack import PackType
from forte.data.ontology.core import (
    Entry,
    BaseLink,
    BaseGroup,
    Grid,
    MultiEntry,
    EntryType,
)
from forte.data.span import Span
from forte.common.constants import (
    BEGIN_INDEX,
    END_INDEX,
    PARENT_TID_INDEX,
    CHILD_TID_INDEX,
    MEMBER_TID_INDEX,
)

__all__ = [
    "Generics",
    "Annotation",
    "Group",
    "Link",
    "MultiPackGeneric",
    "MultiPackGroup",
    "MultiPackLink",
    "Query",
    "SinglePackEntries",
    "MultiPackEntries",
    "AudioAnnotation",
    "ImageAnnotation",
    "Region",
    "Box",
    "BoundingBox",
]

QueryType = Union[Dict[str, Any], np.ndarray]

"""
To create a new top level entry, the following steps are required to
make sure it available across the ontology system:
    1. Create a new top level class that inherits from `Entry` or `MultiEntry`
    2. Add the new class to `SinglePackEntries` or `MultiPackEntries`
    3. Register a new method in `DataStore`: `add_<new_entry>_raw()`
    4. Insert a new conditional branch in `EntryConverter.save_entry_object()`
"""


class Generics(Entry):
    def __init__(self, pack: PackType):
        super().__init__(pack=pack)


@total_ordering
class Annotation(Entry):
    r"""Annotation type entries, such as "token", "entity mention" and
    "sentence". Each annotation has a :class:`~forte.data.span.Span` corresponding to its offset
    in the text.

    Args:
        pack: The container that this annotation
            will be added to.
        begin: The offset of the first character in the annotation.
        end: The offset of the last character in the annotation + 1.
    """

    def __init__(self, pack: PackType, begin: int, end: int):
        self._span: Optional[Span] = None
        self._begin: int = begin
        self._end: int = end
        super().__init__(pack)

    def __getstate__(self):
        r"""For serializing Annotation, we should create Span annotations for
        compatibility purposes.
        """
        self._span = Span(self.begin, self.end)
        state = super().__getstate__()
        state.pop("_begin")
        state.pop("_end")
        return state

    def __setstate__(self, state):
        """
        For de-serializing Annotation, we load the begin, end from Span, for
        compatibility purposes.
        """
        super().__setstate__(state)
        self._begin = self._span.begin
        self._end = self._span.end

    @property
    def span(self) -> Span:
        # Delay span creation at usage.
        if self._span is None:
            self._span = Span(self.begin, self.end)
        return self._span

    @property
    def begin(self):
        r"""Getter function of ``begin``. The function will first try to
        retrieve the begin index from ``DataStore`` in ``self.pack``. If
        this attempt fails, it will directly return the value in ``_begin``.
        """
        try:
            self._begin = self.pack.get_entry_raw(self.tid)[BEGIN_INDEX]
        except KeyError:
            # self.tid not found in DataStore
            pass
        return self._begin

    @begin.setter
    def begin(self, val: int):
        r"""Setter function of ``begin``. The update will also be populated
        into ``DataStore`` in ``self.pack``.
        """
        self._begin = val
        self.pack.get_entry_raw(self.tid)[BEGIN_INDEX] = val

    @property
    def end(self):
        r"""Getter function of ``end``. The function will first try to
        retrieve the end index from ``DataStore`` in ``self.pack``. If
        this attempt fails, it will directly return the value in ``_end``.
        """
        try:
            self._end = self.pack.get_entry_raw(self.tid)[END_INDEX]
        except KeyError:
            # self.tid not found in DataStore
            pass
        return self._end

    @end.setter
    def end(self, val: int):
        r"""Setter function of ``end``. The update will also be populated
        into ``DataStore`` in ``self.pack``.
        """
        self._end = val
        self.pack.get_entry_raw(self.tid)[END_INDEX] = val

    def __eq__(self, other):
        r"""The eq function of :class:`Annotation`.
        By default, :class:`Annotation` objects are regarded as the same if
        they have the same type, span, and are generated by the same component.

        Users can define their own eq function by themselves but this must
        be consistent to :meth:`hash`.
        """
        if other is None:
            return False
        return (type(self), self.begin, self.end) == (
            type(other),
            other.begin,
            other.end,
        )

    def __lt__(self, other):
        r"""To support total_ordering, `Annotation` must implement
        `__lt__`. The ordering is defined in the following way:

        1. If the begin of the annotations are different, the one with larger
           begin will be larger.
        2. In the case where the begins are the same, the one with larger
           end will be larger.
        3. In the case where both offsets are the same, we break the tie using
           the normal sorting of the class name.
        """
        if self.begin == other.begin:
            if self.end == other.end:
                return str(type(self)) < str(type(other))
            return self.end < other.end
        else:
            return self.begin < other.begin

    @property
    def text(self):
        if self.pack is None:
            raise ValueError(
                "Cannot get text because annotation is not "
                "attached to any data pack."
            )
        return self.pack.get_span_text(self.begin, self.end)

    @property
    def index_key(self) -> int:
        return self.tid

    def get(
        self,
        entry_type: Union[str, Type[EntryType]],
        components: Optional[Union[str, Iterable[str]]] = None,
        include_sub_type=True,
    ) -> Iterable[EntryType]:
        """
        This function wraps the :meth:`~forte.data.data_pack.DataPack.get` method to find
        entries "covered" by this annotation. See that method for more information.

        Example:

            .. code-block:: python

                # Iterate through all the sentences in the pack.
                for sentence in input_pack.get(Sentence):
                    # Take all tokens from each sentence created by NLTKTokenizer.
                    token_entries = sentence.get(
                        entry_type=Token,
                        component='NLTKTokenizer')
                    ...

            In the above code snippet, we get entries of type
            :class:`~ft.onto.base_ontology.Token` within
            each ``sentence`` which were generated by ``NLTKTokenizer``. You
            can consider build coverage index between `Token` and `Sentence`
            if this snippet is frequently used.

        Args:
            entry_type: The type of entries requested.
            components: The component (creator)
                generating the entries requested. If `None`, will return valid
                entries generated by any component.
            include_sub_type: whether to consider the sub types of
                the provided entry type. Default `True`.

        Yields:
            Each `Entry` found using this method.

        """
        yield from self.pack.get(entry_type, self, components, include_sub_type)


class Link(BaseLink):
    r"""Link type entries, such as "predicate link". Each link has a parent node
    and a child node.

    Args:
        pack: The container that this annotation
            will be added to.
        parent: the parent entry of the link.
        child: the child entry of the link.
    """
    # this type Any is needed since subclasses of this class will have new types
    ParentType: Any = Entry
    ChildType: Any = Entry

    def __init__(
        self,
        pack: PackType,
        parent: Optional[Entry] = None,
        child: Optional[Entry] = None,
    ):
        self._parent: Optional[int] = None
        self._child: Optional[int] = None
        super().__init__(pack, parent, child)

    # TODO: Can we get better type hint here?
    def set_parent(self, parent: Entry):
        r"""This will set the `parent` of the current instance with given Entry
        The parent is saved internally by its pack specific index key.

        Args:
            parent: The parent entry.
        """
        if not isinstance(parent, self.ParentType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(parent)}"
            )
        self.parent = parent.tid

    def set_child(self, child: Entry):
        r"""This will set the `child` of the current instance with given Entry.
        The child is saved internally by its pack specific index key.

        Args:
            child: The child entry.
        """
        if not isinstance(child, self.ChildType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(child)}"
            )
        self.child = child.tid

    @property
    def parent(self):
        r"""Get ``tid`` of the parent node. To get the object of the parent
        node, call :meth:`get_parent`. The function will first try to
        retrieve the parent ``tid`` from ``DataStore`` in ``self.pack``. If
        this attempt fails, it will directly return the value in ``_parent``.
        """
        try:
            self._parent = self.pack.get_entry_raw(self.tid)[PARENT_TID_INDEX]
        except KeyError:
            # self.tid not found in DataStore
            pass
        return self._parent

    @parent.setter
    def parent(self, val: int):
        r"""Setter function of ``parent``. The update will also be populated
        into ``DataStore`` in ``self.pack``.
        """
        self._parent = val
        self.pack.get_entry_raw(self.tid)[PARENT_TID_INDEX] = val

    @property
    def child(self):
        r"""Get ``tid`` of the child node. To get the object of the child node,
        call :meth:`get_child`. The function will first try to
        retrieve the child ``tid`` from ``DataStore`` in ``self.pack``. If
        this attempt fails, it will directly return the value in ``_child``.
        """
        try:
            self._child = self.pack.get_entry_raw(self.tid)[CHILD_TID_INDEX]
        except KeyError:
            # self.tid not found in DataStore
            pass
        return self._child

    @child.setter
    def child(self, val: int):
        r"""Setter function of ``child``. The update will also be populated
        into ``DataStore`` in ``self.pack``.
        """
        self._child = val
        self.pack.get_entry_raw(self.tid)[CHILD_TID_INDEX] = val

    def get_parent(self) -> Entry:
        r"""Get the parent entry of the link.

        Returns:
             An instance of :class:`~forte.data.ontology.core.Entry` that is the parent of the link.
        """
        if self.pack is None:
            raise ValueError(
                "Cannot get parent because link is not "
                "attached to any data pack."
            )
        if self.parent is None:
            raise ValueError("The parent of this entry is not set.")
        return self.pack.get_entry(self.parent)

    def get_child(self) -> Entry:
        r"""Get the child entry of the link.

        Returns:
             An instance of :class:`~forte.data.ontology.core.Entry` that is the child of the link.
        """
        if self.pack is None:
            raise ValueError(
                "Cannot get child because link is not"
                " attached to any data pack."
            )
        if self.child is None:
            raise ValueError("The child of this entry is not set.")
        return self.pack.get_entry(self.child)


# pylint: disable=duplicate-bases
class Group(BaseGroup[Entry]):
    r"""Group is an entry that represent a group of other entries. For example,
    a "coreference group" is a group of coreferential entities. Each group will
    store a set of members, no duplications allowed.
    """
    MemberType: Type[Entry] = Entry

    def __init__(
        self,
        pack: PackType,
        members: Optional[Iterable[Entry]] = None,
    ):  # pylint: disable=useless-super-delegation
        super().__init__(pack, members)

    def add_member(self, member: Entry):
        r"""Add one entry to the group. The update will be populated to the
        corresponding list in ``DataStore`` of ``self.pack``.

        Args:
            member: One member to be added to the group.
        """
        if not isinstance(member, self.MemberType):
            raise TypeError(
                f"The members of {type(self)} should be "
                f"instances of {self.MemberType}, but got {type(member)}"
            )
        self.pack.get_entry_raw(self.tid)[MEMBER_TID_INDEX].append(member.tid)

    def get_members(self) -> List[Entry]:
        r"""Get the member entries in the group. The function will retrieve
        a list of member entries's ``tid``s from ``DataStore`` and convert them
        to entry object on the fly.

        Returns:
             A set of instances of :class:`~forte.data.ontology.core.Entry`
             that are the members of the group.
        """
        if self.pack is None:
            raise ValueError(
                "Cannot get members because group is not "
                "attached to any data pack."
            )
        member_entries = []
        for m in self.pack.get_entry_raw(self.tid)[MEMBER_TID_INDEX]:
            member_entries.append(self.pack.get_entry(m))
        return member_entries


class MultiPackGeneric(MultiEntry, Entry):
    def __init__(self, pack: PackType):
        super().__init__(pack=pack)


class MultiPackLink(MultiEntry, BaseLink):
    r"""This is used to link entries in a :class:`~forte.data.multi_pack.MultiPack`, which is
    designed to support cross pack linking, this can support applications such
    as sentence alignment and cross-document coreference. Each link should have
    a parent node and a child node. Note that the nodes are indexed by two
    integers, one additional index on which pack it comes from.
    """

    ParentType = Entry
    ChildType = Entry

    def __init__(
        self,
        pack: PackType,
        parent: Optional[Entry] = None,
        child: Optional[Entry] = None,
    ):
        self._parent: Tuple = (None, None)
        self._child: Tuple = (None, None)

        super().__init__(pack)

        if parent is not None:
            self.set_parent(parent)
        if child is not None:
            self.set_child(child)

    @property
    def parent(self):
        r"""Get ``pack_id`` and ``tid`` of the parent node. To get the object
        of the parent node, call :meth:`get_parent`. The function will first
        try to retrieve the parent from ``DataStore`` in ``self.pack``. If
        this attempt fails, it will directly return the value in ``_parent``.
        """
        try:
            self._parent = self.pack.get_entry_raw(self.tid)[PARENT_TID_INDEX]
        except KeyError:
            # self.tid not found in DataStore
            pass
        return self._parent

    @parent.setter
    def parent(self, val: Tuple):
        r"""Setter function of ``parent``. The update will also be populated
        into ``DataStore`` in ``self.pack``.
        """
        self._parent = val
        self.pack.get_entry_raw(self.tid)[PARENT_TID_INDEX] = val

    @property
    def child(self):
        r"""Get ``pack_id`` and ``tid`` of the child node. To get the object
        of the child node, call :meth:`get_child`. The function will first try
        to retrieve the child from ``DataStore`` in ``self.pack``. If
        this attempt fails, it will directly return the value in ``_child``.
        """
        try:
            self._child = self.pack.get_entry_raw(self.tid)[CHILD_TID_INDEX]
        except KeyError:
            # self.tid not found in DataStore
            pass
        return self._child

    @child.setter
    def child(self, val: Tuple):
        r"""Setter function of ``child``. The update will also be populated
        into ``DataStore`` in ``self.pack``.
        """
        self._child = val
        self.pack.get_entry_raw(self.tid)[CHILD_TID_INDEX] = val

    def parent_id(self) -> int:
        """
        Return the ``tid`` of the parent entry.

        Returns:
            The ``tid`` of the parent entry.
        """
        return self.parent[1]

    def child_id(self) -> int:
        """
        Return the ``tid`` of the child entry.

        Returns:
            The ``tid`` of the child entry.
        """
        return self.child[1]

    def parent_pack_id(self) -> int:
        """
        Return the `pack_id` of the parent pack.

        Returns:
            The `pack_id` of the parent pack..
        """
        if self.parent[0] is None:
            raise ValueError("Parent is not set for this link.")
        return self.pack.packs[self.parent[0]].pack_id

    def child_pack_id(self) -> int:
        """
        Return the `pack_id` of the child pack.

        Returns:
            The `pack_id` of the child pack.
        """
        if self.child[0] is None:
            raise ValueError("Child is not set for this link.")
        return self.pack.packs[self.child[0]].pack_id

    def set_parent(self, parent: Entry):
        r"""This will set the `parent` of the current instance with given Entry.
        The parent is saved internally as a tuple: ``pack index`` and
        ``entry.tid``. Pack index is the index of the data pack in the
        multi-pack.

        Args:
            parent: The parent of the link, which is an Entry from a data pack,
                it has access to the pack index and its own ``tid`` in the pack.
        """
        if not isinstance(parent, self.ParentType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(parent)}"
            )
        # fix bug/enhancement #559: using pack_id instead of index
        # self._parent = self.pack.get_pack_index(parent.pack_id), parent.tid
        self.parent = parent.pack_id, parent.tid

    def set_child(self, child: Entry):
        r"""This will set the `child` of the current instance with given Entry.
        The child is saved internally as a tuple: ``pack index`` and
        ``entry.tid``. Pack index is the index of the data pack in the
        multi-pack.

        Args:
            child: The child of the link, which is an Entry from a data pack,
                it has access to the pack index and its own ``tid`` in the pack.
        """

        if not isinstance(child, self.ChildType):
            raise TypeError(
                f"The child of {type(self)} should be an "
                f"instance of {self.ChildType}, but get {type(child)}"
            )
        # fix bug/enhancement #559: using pack_id instead of index
        # self._child = self.pack.get_pack_index(child.pack_id), child.tid
        self.child = child.pack_id, child.tid

    def get_parent(self) -> Entry:
        r"""Get the parent entry of the link.

        Returns:
            An instance of :class:`~forte.data.ontology.core.Entry` that
            is the parent of the link.
        """
        if self._parent is None:
            raise ValueError("The parent of this link is not set.")

        pack_idx, parent_tid = self.parent
        return self.pack.get_subentry(pack_idx, parent_tid)

    def get_child(self) -> Entry:
        r"""Get the child entry of the link.

        Returns:
            An instance of :class:`~forte.data.ontology.core.Entry` that is
            the child of the link.
        """
        if self._child is None:
            raise ValueError("The parent of this link is not set.")

        pack_idx, child_tid = self.child
        return self.pack.get_subentry(pack_idx, child_tid)


# pylint: disable=duplicate-bases
class MultiPackGroup(MultiEntry, BaseGroup[Entry]):
    r"""Group type entries, such as "coreference group". Each group has a set
    of members.
    """
    MemberType: Type[Entry] = Entry

    def __init__(
        self, pack: PackType, members: Optional[Iterable[Entry]] = None
    ):  # pylint: disable=useless-super-delegation
        super().__init__(pack)
        if members is not None:
            self.add_members(members)

    def add_member(self, member: Entry):
        if not isinstance(member, self.MemberType):
            raise TypeError(
                f"The members of {type(self)} should be "
                f"instances of {self.MemberType}, but got {type(member)}"
            )
        self.pack.get_entry_raw(self.tid)[MEMBER_TID_INDEX].append(
            (member.pack_id, member.tid)
        )

    def get_members(self) -> List[Entry]:
        members = []
        for pack_idx, member_tid in self.pack.get_entry_raw(self.tid)[
            MEMBER_TID_INDEX
        ]:
            members.append(self.pack.get_subentry(pack_idx, member_tid))
        return members


@dataclass
class Query(Generics):
    r"""An entry type representing queries for information retrieval tasks.

    Args:
        pack: Data pack reference to which this query will be added
    """
    value: Optional[QueryType]
    results: Dict[str, float]

    def __init__(self, pack: PackType):
        super().__init__(pack)
        self.value: Optional[QueryType] = None
        self.results: Dict[str, float] = {}

    def add_result(self, pid: str, score: float):
        """
        Set the result score for a particular pack (based on the pack id).

        Args:
            pid: the pack id.
            score: the score for the pack

        Returns:
            None
        """
        self.results[pid] = score

    def update_results(self, pid_to_score: Dict[str, float]):
        r"""Updates the results for this query.

        Args:
             pid_to_score: A dict containing pack id -> score mapping
        """
        self.results.update(pid_to_score)


@total_ordering
class AudioAnnotation(Entry):
    r"""AudioAnnotation type entries, such as "recording" and "audio utterance".
    Each audio annotation has a :class:`~forte.data.span.Span` corresponding to its offset
    in the audio. Most methods in this class are the same as the ones in
    :class:`Annotation`, except that it replaces property `text` with `audio`.

    Args:
        pack: The container that this audio annotation
            will be added to.
        begin: The offset of the first sample in the audio annotation.
        end: The offset of the last sample in the audio annotation + 1.
    """

    def __init__(self, pack: PackType, begin: int, end: int):
        self._span: Optional[Span] = None
        self._begin: int = begin
        self._end: int = end
        super().__init__(pack)

    @property
    def audio(self):
        if self.pack is None:
            raise ValueError(
                "Cannot get audio because annotation is not "
                "attached to any data pack."
            )
        return self.pack.get_span_audio(self.begin, self.end)

    def __getstate__(self):
        r"""For serializing AudioAnnotation, we should create Span annotations
        for compatibility purposes.
        """
        self._span = Span(self.begin, self.end)
        state = super().__getstate__()
        state.pop("_begin")
        state.pop("_end")
        return state

    def __setstate__(self, state):
        """
        For de-serializing AudioAnnotation, we load the begin, end from Span,
        for compatibility purposes.
        """
        super().__setstate__(state)
        self._begin = self._span.begin
        self._end = self._span.end

    @property
    def span(self) -> Span:
        # Delay span creation at usage.
        if self._span is None:
            self._span = Span(self.begin, self.end)
        return self._span

    @property
    def begin(self):
        r"""Getter function of ``begin``. The function will first try to
        retrieve the begin index from ``DataStore`` in ``self.pack``. If
        this attempt fails, it will directly return the value in ``_begin``.
        """
        try:
            self._begin = self.pack.get_entry_raw(self.tid)[BEGIN_INDEX]
        except KeyError:
            # self.tid not found in DataStore
            pass
        return self._begin

    @begin.setter
    def begin(self, val: int):
        r"""Setter function of ``begin``. The update will also be populated
        into ``DataStore`` in ``self.pack``.
        """
        self._begin = val
        self.pack.get_entry_raw(self.tid)[BEGIN_INDEX] = val

    @property
    def end(self):
        r"""Getter function of ``end``. The function will first try to
        retrieve the end index from ``DataStore`` in ``self.pack``. If
        this attempt fails, it will directly return the value in ``_end``.
        """
        try:
            self._end = self.pack.get_entry_raw(self.tid)[END_INDEX]
        except KeyError:
            # self.tid not found in DataStore
            pass
        return self._end

    @end.setter
    def end(self, val: int):
        r"""Setter function of ``end``. The update will also be populated
        into ``DataStore`` in ``self.pack``.
        """
        self._end = val
        self.pack.get_entry_raw(self.tid)[END_INDEX] = val

    def __eq__(self, other):
        r"""The eq function of :class:`AudioAnnotation`.
        By default, :class:`AudioAnnotation` objects are regarded as the same if
        they have the same type, span, and are generated by the same component.

        Users can define their own eq function by themselves but this must
        be consistent to :meth:`hash`.
        """
        if other is None:
            return False
        return (type(self), self.begin, self.end) == (
            type(other),
            other.begin,
            other.end,
        )

    def __lt__(self, other):
        r"""To support total_ordering, `AudioAnnotation` must implement
        `__lt__`. The ordering is defined in the following way:

        1. If the begin of the audio annotations are different, the one with
           larger begin will be larger.
        2. In the case where the begins are the same, the one with larger
           end will be larger.
        3. In the case where both offsets are the same, we break the tie using
           the normal sorting of the class name.
        """
        if self.begin == other.begin:
            if self.end == other.end:
                return str(type(self)) < str(type(other))
            return self.end < other.end
        else:
            return self.begin < other.begin

    @property
    def index_key(self) -> int:
        return self.tid

    def get(
        self,
        entry_type: Union[str, Type[EntryType]],
        components: Optional[Union[str, Iterable[str]]] = None,
        include_sub_type=True,
    ) -> Iterable[EntryType]:
        """
        This function wraps the :meth:`~forte.data.data_pack.DataPack.get()` method to find
        entries "covered" by this audio annotation. See that method for more
        information. For usage details, refer to
        :meth:`forte.data.ontology.top.Annotation.get()`.

        Args:
            entry_type: The type of entries requested.
            components: The component (creator)
                generating the entries requested. If `None`, will return valid
                entries generated by any component.
            include_sub_type: whether to consider the sub types of
                the provided entry type. Default `True`.

        Yields:
            Each `Entry` found using this method.

        """
        yield from self.pack.get(entry_type, self, components, include_sub_type)


class ImageAnnotation(Entry):
    def __init__(self, pack: PackType, image_payload_idx: int = 0):
        """
        ImageAnnotation type entries, such as "edge" and "bounding box".
        Each ImageAnnotation has a ``image_payload_idx`` corresponding to its
        image representation in the payload array.

        Args:
            pack: The container that this image annotation
                will be added to.
            image_payload_idx: the index of the image payload. If it's not set,
                it defaults to 0 which means it will load the first image payload.
        """
        self._image_payload_idx = image_payload_idx
        super().__init__(pack)

    @property
    def image_payload_idx(self) -> int:
        return self._image_payload_idx

    @property
    def image(self):
        if self.pack is None:
            raise ValueError(
                "Cannot get image because image annotation is not "
                "attached to any data pack."
            )
        return self.pack.get_image_array(self._image_payload_idx)

    @property
    def max_x(self):
        return self.image.shape[1] - 1

    @property
    def max_y(self):
        return self.image.shape[0] - 1

    def __eq__(self, other):
        if other is None:
            return False
        return self.image_payload_idx == other.image_payload_idx


class Region(ImageAnnotation):
    """
    A region class associated with an image payload.

    Args:
        pack: the container that this ``Region`` will be added to.
        image_payload_idx: the index of the image payload. If it's not set,
            it defaults to 0 which meaning it will load the first image payload.
    """

    def __init__(self, pack: PackType, image_payload_idx: int = 0):
        super().__init__(pack, image_payload_idx)
        if image_payload_idx is None:
            self._image_payload_idx = 0
        else:
            self._image_payload_idx = image_payload_idx

    def compute_iou(self, other) -> int:
        intersection = np.sum(np.logical_and(self.image, other.image))
        union = np.sum(np.logical_or(self.image, other.image))
        return intersection / union


class Box(Region):
    """
    A box class with a reference point which is the box center and a box
    configuration.

    Box of a certain size (height, width), as a regular shape, can be located
    only when it has a reference point (box center here).

    There are several use cases for a box:
        1. When we use a box standalone, we need cy, cx to be set. The offset between the box center and the grid cell center is not used.
        2. When we represent a ground truth box, the box center and its shape
        are given. We can compute the offset between the box center and the grid cell center.
        3. When we predict a box, we will have the predicted box shape (height, width) and the offset between the box center and the grid cell center, then we can compute the box center.

    Based on the use cases, there are two important class conditions:
        1. Whether the box center is set.
        2. Whether it's associated with a grid. (It might or might not depending on the box use cases)

    Note: all indices are zero-based and counted from top left corner of
    image. But the unit could be a pixel or a grid cell.

    Args:
        pack: the container that this ``Box`` will be added to.

        height: the height of the box, the unit is one pixel.
        width: the width of the box, the unit is one pixel.
        cy: the row index of the box center in the image array,
            the unit is one pixel. If not set, the box center
            will be set to the center of image (half height of the image).
        cx: the column index of the box center in the image array,
            the unit is one pixel. If not set, the box center
            will be set to the center of image (half width of the image).
        image_payload_idx: the index of the image payload. If it's not set,
            it defaults to 0 which meaning it will load the first image payload.
    """

    def __init__(
        self,
        pack: PackType,
        height: int,
        width: int,
        cy: Optional[int] = None,
        cx: Optional[int] = None,
        image_payload_idx: int = 0,
    ):
        # assume Box is associated with Grid
        super().__init__(pack, image_payload_idx)

        # We don't initialize the grid cell center/offset during the class
        # initialization because we cannot pass the grid cell to the constructor.
        # Instead, we initialize the grid cell center when we set the grid cell.
        if height > 0 and width > 0:
            self._height = height
            self._width = width
        else:
            raise ValueError(
                f"Box height({height}) and width({width}) must be positive."
            )

        # TODO: implement the upper bound check for the box height and width
        # after the payload PR https://github.com/asyml/forte/pull/828 is merged

        if cy is not None and cx is not None:
            self._check_center_validity(cy, cx)
            self._cy = cy
            self._cx = cx
        # TODO: implement the upper bound check for the box height and width
        # after the payload PR https://github.com/asyml/forte/pull/828 is merged
        self._cy = cy
        self._cx = cx
        # intialize the grid cell center/offset to None
        # they must be set later if the ``Box`` use case is associated with Grid
        self._is_grid_associated = False
        self._cy_offset: Optional[int] = None
        self._cx_offset: Optional[int] = None
        self._grid_cy: Optional[int] = None
        self._grid_cx: Optional[int] = None

    def _check_center_validity(
        self, cy: Optional[int], cx: Optional[int]
    ) -> bool:
        """
        Check whether the box center is valid.

        Args:
            cy: the row index of the box center in the image array.
            cx: the column index of the box center in the image array.

        Returns:
            True if the box center is valid, False otherwise.
        """
        if cy is None or cx is None:
            raise ValueError(
                "Box center cy, cx must be set." "Currently they are None."
            )
        # TODO: implement the upper bound check for the box center
        # after the payload PR https://github.com/asyml/forte/pull/828 is merged
        # if cy >= self.max_y or cx >= self.max_x:
        #     raise ValueError(f"Box center({cy}, {cx}) must be less than max_y({self.max_y}) and max_x({self.max_x}).")
        if cy < self._height / 2 or cx < self._width / 2:
            raise ValueError(
                f"Box center({cy}, {cx}) must be greater than half height({self._height/2}) and half width({self._width/2}) respectively."
            )

    def set_center(self, cy: int, cx: int):
        """
        Set the center(reference point) of the Box.

        Args:
            cy: the row index of the box center in the image array,
                the unit is one pixel.
            cx: the column index of the box center in the image array,
                the unit is one pixel.
        """
        self._check_center_validity(cy, cx)
        self._cy = cy
        self._cx = cx

    def set_grid_cell_center(
        self, grid: Grid, grid_h_idx: int, grid_w_idx: int
    ):
        """
        Set the center of a grid cell that the Box is associated with.

        Args:
            grid: the grid that the box is associated with.
            grid_h_idx: the row index of the grid cell center in the image
                array, the unit is one grid cell.
            grid_w_idx: the column index of the grid cell center in the image
                array, the unit is one grid cell.
        """
        self._is_grid_associated = True
        # given a grid cell, compute its center
        self._grid_cy, self._grid_cx = grid.get_grid_cell_center(
            grid_h_idx, grid_w_idx
        )

    @property
    def is_grid_associated(self) -> bool:
        """
        A boolean indicating whether the box is associated with a grid.

        Returns:
            True if the box is associated with a grid, False otherwise.
        """
        return self._is_grid_associated

    @property
    def grid_cy(self) -> int:
        """
        The row index of the grid cell center in the image array.

        Raises:
            ValueError: if the box is not associated with a grid.

        Returns:
            The row index of the grid cell center in the image array.
        """
        if self.is_grid_associated:
            return self._grid_cy
        else:
            raise ValueError(
                "The box is not associated with a grid."
                "Therefore, there is no grid cell center."
            )
        return self._grid_cy

    @property
    def grid_cx(self) -> int:
        """
        The column index of the grid cell center in the image array.

        Raises:
            ValueError: if the box is not associated with a grid.

        Returns:
            The column index of the grid cell center in the image array.
        """

        if self.is_grid_associated:
            return self._grid_cx
        else:
            raise ValueError(
                "The box is not associated with a grid."
                "Therefore, there is no grid cell center."
            )

    @property
    def grid_cell_center(self) -> Tuple[int, int]:
        """
        The center of the grid cell that the Box is associated with.

        Raises:
            ValueError: if the box is not associated with a grid.

        Returns:
            Tuple[int, int]: the center of the grid cell that the Box is associated with.
        """
        if self.is_grid_associated:
            return self.grid_cy, self.grid_cx
        else:
            raise ValueError(
                "The box is not associated with a grid."
                "Therefore, there is no grid cell center."
            )

    def set_offset(self, cy_offset: int, cx_offset: int):
        """
        Set the offset of the box center from the grid cell center.

        Args:
            cy_offset: the row index of the box center offset from the grid cell
                center in the image array, the unit is one pixel.
            cx_offset: the column index of the box center offset from the grid
                cell center in the image array, the unit is one pixel.
        """
        self._cy_offset = cy_offset
        self._cx_offset = cx_offset

    @property
    def offset(self):
        """
        The offset of the box center from the grid cell center.

        Returns:
            the offset of the box center from the grid cell
        """
        return self._cy_offset, self._cx_offset

    @property
    def cy_offset(self) -> Optional[int]:
        """
        The row index difference between the box center and the grid cell.

        Returns:
            The row index difference between the box center and the grid cell.
        """
        if self._cy_offset:
            return self._cy_offset
        if self._is_grid_associated and self._is_box_center_set():
            self._cy_offset = self._cy - self._grid_cy
            return self._cy_offset
        else:
            self._offset_condition_check()

    @property
    def cx_offset(self) -> Optional[int]:
        """
        The column index difference between the box center and the grid cell

        Returns:
            The column index difference between the box center and the grid cell
        """
        if self._cx_offset:
            return self._cx_offset
        if self._is_grid_associated and self._is_box_center_set():
            self._cx_offset = self._cx - self._grid_cx
            return self._cx_offset
        else:
            self._offset_condition_check()

    @property
    def cy(self) -> int:
        """
        Compute and return row index of the box center in the image array.
        It returns the row index of the box center in the image array directly
        if the box center is set.
        Otherwise, if it computes and sets the box center y coordinate when the box is both associated with a grid and the
        offset is set.

        Returns:
            The row index of the box center in the image array.
        """
        if self._cy is not None:
            return self._cy
        else:
            # if cy computation condition is met, then cy is set
            if self._is_grid_associated and self._is_offset_set():
                self._cy = self._grid_cy + self._cy_offset
                return self._cy
            else:
                self._center_condition_check()

    @property
    def cx(self) -> int:
        """
        The column index of the box center in the image array.
        It returns the column index of the box center in the image array
        directly if the box center is set.
        Otherwise, if it computes and sets the box center x coordinate when the box is both associated with a grid and the
        offset is set.

        Returns:
            The column index of the box center in the image array.
        """
        if self._cx is not None:
            return self._cx
        else:
            # if cx computation condition is met, then cx is set
            if self._is_grid_associated and self._is_offset_set():
                self._cx = self._grid_cx + self._cx_offset
                return self._cx
            else:
                self._center_condition_check()

    @property
    def box_center(self) -> Tuple[int, int]:
        """
        Get the box center by using the property function cy() and cx().
        If box center is not set not computable, it raises a ``ValueError``

        Returns:
            The box center in a ``Tuple`` format.
        """
        return (self.cy, self.cx)

    @property
    def corners(self) -> Tuple[int, int, int, int]:
        """
        Compute and return the corners of the box.

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The corners of the box in a ``Tuple`` format.
        """
        if self._is_box_center_set():
            return tuple(
                [
                    (self._cy + h_offset, self._cx + w_offset)
                    for h_offset in [-self._height // 2, self._height // 2]
                    for w_offset in [-self._width // 2, self._width // 2]
                ]
            )
        else:
            raise ValueError(
                "The box center is not set so the box corners"
                " cannot be calculated."
                "Please set the box center first or make sure "
                "box center computation condition is met."
            )

    @property
    def box_min_x(self) -> int:
        """
        Compute the minimum x coordinate of the box.

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The minimum x coordinate of the box.
        """
        if self._is_box_center_set():
            return max(self._cx - round(0.5 * self._width), 0)
        else:
            raise ValueError(
                "The box center is not set so the box min x"
                " coordinate cannot be calculated."
                "Please set the box center first or make sure "
                "box center computation condition is met."
            )

    @property
    def box_max_x(self) -> int:
        """
        Compute the maximum x coordinate of the box.

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The maximum x coordinate of the box.
        """
        if self._is_box_center_set():
            # TODO: check the upper bound
            return self._cx + self._width // 2
        else:
            raise ValueError(
                "The box center is not set so the box max x"
                " coordinate cannot be calculated."
                "Please set the box center first or make sure "
                "box center computation condition is met."
            )

    @property
    def box_min_y(self) -> int:
        """
        Compute the minimum y coordinate of the box.

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The minimum y coordinate of the box.
        """
        if self._is_box_center_set():
            return max(self._cy - round(0.5 * self._height), 0)
        else:
            raise ValueError(
                "The box center is not set so the box min y"
                " coordinate cannot be calculated."
                "Please set the box center first or make sure "
                "box center computation condition is met."
            )

    @property
    def box_max_y(self) -> int:
        """
        Compute the maximum y coordinate of the box.

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The maximum y coordinate of the box.
        """
        if self._is_box_center_set():
            # TODO: add upper bound
            return self._cy + round(0.5 * self._height)
        else:
            raise ValueError(
                "The box center is not set so the box max y"
                " coordinate cannot be calculated."
                "Please set the box center first or make sure "
                "box center computation condition is met."
            )

    @property
    def area(self) -> int:
        """
        Compute the area of the box.

        Returns:
            The area of the box.
        """
        return self._height * self._width

    def is_overlapped(self, other) -> bool:
        """
        A function checks whether two boxes are overlapped(two box area have
        intersections).

        Note: in edges cases where two bounding boxes' boundaries share the
        same line segment/corner in the image array, it won't be considered
        overlapped.

        Args:
            other: the other ``Box`` object to compared to.

        Returns:
            True if the two boxes are overlapped, False otherwise.
        """
        if not isinstance(other, Box):
            raise ValueError(
                "The other object to check overlapping with is"
                " not a Box object."
                "You need to check the type of the other object."
            )

        # If one box is on left side of other
        if self.box_min_x > other.box_max_x or other.box_min_x > self.box_max_x:
            return False

        # If one box is above other
        if self.box_min_y > other.box_max_y or other.box_min_y > self.box_max_y:
            return False
        return True

    def compute_iou(self, other) -> float:
        """
        A function computes iou(intersection over union) between two boxes.
        It overwrites the ``compute_iou`` function in it's parent class
        ``Region``.

        Args:
            other: the other ``Box`` object to be computed with.

        Returns:
            A float value which is (intersection area/ union area) between two
            boxes.
        """
        if not isinstance(other, Box):
            raise ValueError(
                "The other object to compute iou with is"
                " not a Box object."
                "You need to check the type of the other object."
            )

        if not self.is_overlapped(other):
            return 0
        box_x_diff = min(
            abs(other.box_max_x - self.box_min_x),
            abs(other.box_min_x - self.box_max_x),
        )
        box_y_diff = min(
            abs(other.box_max_y - self.box_min_y),
            abs(other.box_min_y - self.box_max_y),
        )
        intersection = box_x_diff * box_y_diff
        union = self.area + other.area - intersection
        return intersection / union

    def _offset_condition_check(self):
        """
        When the the offset is not set, this function checks the reason the offset cannot be computed and raises the corresponding error.

        Raises:
            ValueError: if the grid cell is not associated with the box and
                the box center is not set.
            ValueError: if the grid cell is not associated with the box.
            ValueError: if the box center is not set.
        """
        result_msg = "Hence, the offset of the box center from the grid cell center cannot be computed."
        if not self._is_grid_associated and not self._is_box_center_set():
            raise ValueError(
                "The box center is not set and the grid cell center is not set."
                + result_msg
            )
        elif not self._is_grid_associated:
            raise ValueError("The grid cell center is not set." + result_msg)
        elif not self._is_box_center_set():
            raise ValueError("The box center is not set." + result_msg)

    def _center_condition_check(self):
        """
        When the the center is not set, this function checks the reason the box center cannot be computed and raises the corresponding error.

        Raises:
            ValueError: if the box center is not set and the grid cell center
                is not set.
            ValueError: if the grid cell center is not set.
            ValueError: if the box center is not set.
        """
        result_msg = "Hence, the position of the box center cannot be computed."
        if not self._is_grid_associated and not self._is_offset_set():
            raise ValueError(
                "The box center is not set and the grid cell center is not set."
                + result_msg
            )
        elif not self._is_grid_associated:
            raise ValueError("The grid cell center is not set." + result_msg)
        elif not self._is_offset_set():
            raise ValueError("The offset is not set." + result_msg)

    def _is_box_center_set(self) -> bool:
        """
        A function checks whether the box center is set.

        Returns:
            True if the box center is set, False otherwise.
        """
        return self._cy is not None and self._cx is not None

    def _is_offset_set(self) -> bool:
        """
        A function checks whether the offset of the box center from the grid
        is set.

        Returns:
            True if the offset is set, False otherwise.
        """
        return self._cy_offset is not None and self._cx_offset is not None


class BoundingBox(Box):
    """
    A bounding box class that associates with image payload and grid and
    has a configuration of height and width.

    Note: all indices are zero-based and counted from top left corner of
    the image/grid.

    Args:
        pack: The container that this BoundingBox will
            be added to.
        image_payload_idx: the index of the image payload. If it's not set,
            it defaults to 0 which means it will load the first image payload.
        height: the height of the bounding box, the unit is one image array
            entry.
        width: the width of the bounding box, the unit is one pixel.
        grid_height: the height of the associated grid, the unit is one grid
            cell.
        grid_width: the width of the associated grid, the unit is one grid
            cell.
        grid_cell_h_idx: the height index of the associated grid cell in
            the grid, the unit is one grid cell.
        grid_cell_w_idx: the width index of the associated grid cell in
            the grid, the unit is one grid cell.
    """

    def __init__(
        self,
        pack: PackType,
        height: int,
        width: int,
        image_payload_idx: int = 0,
    ):

        self._is_grid_associated = False
        super().__init__(
            pack,
            height,
            width,
            image_payload_idx,
        )


SinglePackEntries = (
    Link,
    Group,
    Annotation,
    Generics,
    AudioAnnotation,
    ImageAnnotation,
)
MultiPackEntries = (MultiPackLink, MultiPackGroup, MultiPackGeneric)
