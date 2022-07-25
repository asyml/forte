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
from enum import IntEnum
from functools import total_ordering
from typing import (
    Optional,
    Sequence,
    Tuple,
    Type,
    Any,
    Dict,
    Union,
    Iterable,
    List,
)
import numpy as np

from forte.data.modality import Modality
from forte.data.base_pack import PackType
from forte.data.ontology.core import (
    Entry,
    BaseLink,
    BaseGroup,
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
    "Payload",
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
        ImageAnnotation type entries, such as "edge" and "box".
        Each ImageAnnotation has a ``image_payload_idx`` corresponding to its
        image representation in the payload array.

        Args:
            pack: The container that this image annotation
                will be added to.
            image_payload_idx: the index of the image payload in the DataPack's
                image payload list.
                If it's not set, it defaults to 0 which means it will load the
                first image payload.
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
        return self.pack.get_payload_data_at(
            Modality.Image, self._image_payload_idx
        )

    @property
    def max_x(self):
        return self._image_width - 1

    @property
    def max_y(self):
        return self._image_height - 1

    @property
    def image_shape(self):
        return (self._image_height, self._image_width)

    def set_image_shape(self, width, height):
        """
        This function is used to set the shape of the image.

        Args:
            width: the width of the image. The unit is pixel.
            height: the height of the image. The unit is pixel.
        """
        self._image_width = (  # pylint: disable=attribute-defined-outside-init
            width
        )
        self._image_height = (  # pylint: disable=attribute-defined-outside-init
            height
        )

    def __eq__(self, other):
        if other is None:
            return False
        return self.image_payload_idx == other.image_payload_idx


class Region(ImageAnnotation):
    """
    A region class associated with an image payload.

    Args:
        pack: the container that this ``Region`` will be added to.
        image_payload_idx: the index of the image payload in the DataPack's
            image payload list.
            If it's not set,
            it defaults to 0 which meaning it will load the first image payload.
    """

    def __init__(self, pack: PackType, image_payload_idx: int = 0):
        super().__init__(pack, image_payload_idx)
        if image_payload_idx is None:
            self._image_payload_idx = 0
        else:
            self._image_payload_idx = image_payload_idx

    def compute_iou(self, other) -> float:
        intersection = np.sum(np.logical_and(self.image, other.image))
        union = np.sum(np.logical_or(self.image, other.image))
        return intersection / union


class Box(Region):
    """
    A box class with a center position and a box configuration.

    Note: all indices are zero-based and counted from top left corner of
    image.

    Args:
        pack: the container that this ``Box`` will be added to.
        tl_point: the indices of top left point of the box
            [row index, column index], the unit is one pixel.
        br_point: the indices of bottom right point of the box
            [row index, column index], the unit is one pixel.
        image_payload_idx: the index of the image payload in the DataPack's
            image payload list. If it's not set,
            it defaults to 0 which meaning it will load the first image payload.
    """

    def __init__(
        self,
        pack: PackType,
        tl_point: List[int],
        br_point: List[int],
        image_payload_idx: int = 0,
    ):
        super().__init__(pack, image_payload_idx)
        if tl_point[0] < 0 or tl_point[1] < 0:
            raise ValueError(
                f"input parameter top left point indices ({tl_point}) must"
                "be non-negative"
            )
        if br_point[0] < 0 or br_point[1] < 0:
            raise ValueError(
                f"input parameter bottom right point indices ({br_point}) must"
                "be non-negative"
            )
        if tl_point[0] >= br_point[0]:
            raise ValueError(
                f"top left point y coordinate({tl_point[0]}) must be less than"
                f" bottom right y coordinate({br_point[0]})"
            )
        if tl_point[1] >= br_point[1]:
            raise ValueError(
                f"top left point x coordinate({tl_point[1]}) must be less than"
                f" bottom right x coordinate({br_point[1]})"
            )

        self._y0, self._x0 = tl_point
        self._y1, self._x1 = br_point
        self._cy = round((self._y0 + self._y1) / 2)
        self._cx = round((self._x0 + self._x1) / 2)
        self._height = self._y1 - self._y0
        self._width = self._x1 - self._x0

    @classmethod
    def init_from_center_n_shape(
        cls,
        pack: PackType,
        cy: int,
        cx: int,
        height: int,
        width: int,
        image_payload_idx: int = 0,
    ):
        """
        A class method to initialize a ``Box`` from a box's center position and
        shape.

        Note: all indices are zero-based and counted from top left corner of
        image.

        Args:
            pack: the container that this ``Box`` will be added to.
            cy: the row coordinate of the box's center, the unit is one pixel.
            cx: the column coordinate of the box's center, the unit is one pixel.
            height: the height of the box, the unit is one pixel.
            width: the width of the box, the unit is one pixel.
            image_payload_idx: the index of the image payload in the DataPack's
                image payload list. If it's not set, it defaults to 0 which
                meaning it will load the first image payload.

        Returns:
            A ``Box`` instance.
        """
        # center location
        return cls(
            pack,
            [cy - round(height / 2), cx - round(width / 2)],
            [cy - round(height / 2) + height, cx - round(width / 2) + width],
            image_payload_idx,
        )

    def compute_iou(self, other) -> float:
        """
        A function computes iou(intersection over union) between two boxes
        (unit: pixel).
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

    @property
    def center(self):
        return (self._cy, self._cx)

    @property
    def corners(self):
        """
        Get corners of box.
        """
        return [
            (self._y0, self._x0),
            (self._y0, self._x1),
            (self._y1, self._x0),
            (self._y1, self._x1),
        ]

    @property
    def box_min_x(self):
        return self._x0

    @property
    def box_max_x(self):
        return min(self._x1, self.max_x)

    @property
    def box_min_y(self):
        return self._y0

    @property
    def box_max_y(self):
        return min(self._y1, self.max_y)

    @property
    def area(self):
        return self._height * self._width

    def is_overlapped(self, other):
        """
        A function checks whether two boxes are overlapped(two box area have
        intersections).

        Note: in edges cases where two boxes' boundaries share the
        same line segment/corner in the image array, it won't be considered
        overlapped.

        Args:
            other: the other ``Box`` object to compared to.

        Returns:
            A boolean value indicating whether there is overlapped.
        """
        # If one box is on left side of other
        if self.box_min_x > other.box_max_x or other.box_min_x > self.box_max_x:
            return False

        # If one box is above other
        if self.box_min_y > other.box_max_y or other.box_min_y > self.box_max_y:
            return False
        return True


class Payload(Entry):
    """
    A payload class that holds data cache of one modality and its data source uri.

    Args:
        pack: The container that this `Payload` will
            be added to.
        modality: modality of the payload such as text, audio and image.
        payload_idx: the index of the payload in the DataPack's
            image payload list of the same modality. For example, if we
            instantiate a ``TextPayload`` inherited from ``Payload``, we assign
            the payload index in DataPack's text payload list.
        uri: universal resource identifier of the data source. Defaults to None.

    Raises:
        ValueError: raised when the modality is not supported.
    """

    def __init__(
        self,
        pack: PackType,
        payload_idx: int = 0,
        uri: Optional[str] = None,
    ):
        from ft.onto.base_ontology import (  # pylint: disable=import-outside-toplevel
            TextPayload,
            AudioPayload,
            ImagePayload,
        )

        # since we cannot pass different modality from generated ontology, and
        # we don't want to import base ontology in the header of the file
        # we import it here.
        if isinstance(self, TextPayload):
            self._modality = Modality.Text
        elif isinstance(self, AudioPayload):
            self._modality = Modality.Audio
        elif isinstance(self, ImagePayload):
            self._modality = Modality.Image
        else:
            supported_modality = [enum.name for enum in Modality]
            raise ValueError(
                f"The given modality {self._modality.name} is not supported. "
                f"Currently we only support {supported_modality}"
            )
        self._payload_idx: int = payload_idx
        self._uri: Optional[str] = uri

        super().__init__(pack)
        self._cache: Union[str, np.ndarray] = ""
        self.replace_back_operations: Sequence[Tuple] = []
        self.processed_original_spans: Sequence[Tuple] = []
        self.orig_text_len: int = 0

    def get_type(self) -> type:
        """
        Get the class type of the payload class. For example, suppose a
        ``TextPayload`` inherits this ``Payload`` class, ``TextPayload`` will be
        returned.

        Returns:
            the type of the payload class.
        """
        return type(self)

    @property
    def cache(self) -> Union[str, np.ndarray]:
        return self._cache

    @property
    def modality(self) -> IntEnum:
        """
        Get the modality of the payload class.

        Returns:
            the modality of the payload class in ``IntEnum`` format.
        """
        return self._modality

    @property
    def modality_name(self) -> str:
        """
        Get the modality of the payload class in str format.

        Returns:
            the modality of the payload class in str format.
        """
        return self._modality.name

    @property
    def payload_index(self) -> int:
        return self._payload_idx

    @property
    def uri(self) -> Optional[str]:
        return self._uri

    def set_cache(self, data: Union[str, np.ndarray]):
        """
        Load cache data into the payload.

        Args:
            data: data to be set in the payload. It can be str for text data or
                numpy array for audio or image data.
        """
        self._cache = data

    def set_payload_index(self, payload_index: int):
        """
        Set payload index for the DataPack.

        Args:
            payload_index: a new payload index to be set.
        """
        self._payload_idx = payload_index

    def __getstate__(self):
        r"""
        Convert ``_modality`` ``Enum`` object to str format for serialization.
        """
        # TODO: this function will be removed since
        # Entry store is being integrated into DataStore
        state = self.__dict__.copy()
        state["_modality"] = self._modality.name
        return state

    def __setstate__(self, state):
        r"""
        Convert ``_modality`` string to ``Enum`` object for deserialization.
        """
        # TODO: this function will be removed since
        # Entry store is being integrated into DataStore
        self.__dict__.update(state)
        self._modality = getattr(Modality, state["_modality"])


SinglePackEntries = (
    Link,
    Group,
    Annotation,
    Generics,
    AudioAnnotation,
    ImageAnnotation,
    Payload,
)
MultiPackEntries = (MultiPackLink, MultiPackGroup, MultiPackGeneric)
