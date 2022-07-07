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
        ImageAnnotation type entries, such as "edge" and "bounding box".
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

    def compute_iou(self, other) -> float:
        """
        Compute the IoU between this region and another region.

        Args:
            other: Another region object.

        Returns:
            the IoU between this region and another region as a float.
        """
        if not isinstance(other, Region):
            raise TypeError("other must be a Region object")
        intersection = np.sum(np.logical_and(self.image, other.image))
        union = np.sum(np.logical_or(self.image, other.image))
        return intersection / union


class Box(Region):
    # pylint: disable=too-many-public-methods
    """
    A box class with a reference point which is the box center and a box
    configuration.

    Given a box with shape parameters (height, width), we want to locate its
    four corners (top-left, top-right, bottom-left, bottom-right). We need
    to know the locaton of the center of the box.

    Generally, we can either use box standalone or as a bounding box in object
    detection tasks. In the later case, we need to consider its association with
    a grid as the task is performed on each grid cell.
    There are several use cases for a box:
        1. When we use a box standalone, we need the box center to be set.
        The offset between the box center and the grid cell center is not used.
        2. When we represent a ground truth box, the box center and shape are
        given. If we want to compute loss, its grid cell center is required to
        compute the offset between the box center and the
        grid cell center.
        3. When we predict a box, we will have the predicted box shape (height,
        width) and the offset between the box center and the grid cell center,
        then we can compute the box center.

    .. code-block:: python
        pack = DataPack("box examples")
        # create a box
        simple_box = Box(pack, 640, 480, 320, 240)
        # create a bounding box at with its center at (320, 240)
        gt_bbx = Box(pack, 640, 480, 320, 240)
        # create a predicted bounding box without known its center
        # and compute its center from the offset and the grid center
        predicted_bbx = Box(pack, 640, 480)
        predicted_bbx.set_offset(305, 225)
        b.set_grid_cell_center(Grid(64, 48, 640, 480), 1, 1)
        print(b.cy, b.cx) # it prints (320, 240)


    For example, in the object detection task, dataset label contains a ground
    truth box (box shape and box center).
    The inference pipeline is that given a grid cell, we make a prediction of a
    bounding box (box shape and box offset from the grid cell center).
        1. If we want to locate the predicted box, we compute the box center
        based on the second use case.
        2. If we want to compute the loss, we need values for the center, shape
        and offset for both boxes. And we need to compute the offset between
        the box center and the grid cell center based on the third use case.

    A more detailed explanation can be found in the following blog:
    https://towardsdatascience.com/yolo2-walkthrough-with-examples-e40452ca265f

    Based on the use cases, there are two important class conditions:
        1. Whether the box center is set.
        2. Whether it's associated with a grid. (It might or might not
        depending on the box use cases)

    Note: all indices are zero-based and counted from top left corner of
    image. But the unit could be a pixel or a grid cell.

    Args:
        pack: the container that this ``Box`` will be added to.

        height: the height of the box, the unit is one pixel.
        width: the width of the box, the unit is one pixel.
        cy: the row index of the box center in the image array,
            the unit is one pixel. If not set, it defaults to None.
        cx: the column index of the box center in the image array,
            the unit is one pixel. If not set, it defaults to None.
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
        super().__init__(pack, image_payload_idx)

        self._image_height, self._image_width = self.pack.get_payload_data_at(
            Modality.Image, image_payload_idx
        ).shape[-2:]
        # We don't initialize the grid cell center/offset during the class
        # initialization because we cannot pass the grid cell to the constructor.
        # Instead, we initialize the grid cell center when we set the grid cell.
        if not (height > 0 and width > 0):
            raise ValueError(
                f"Box height({height}) and width({width}) must be positive."
            )
        if not (height < self._image_height and width < self._image_width):
            raise ValueError(
                f"Box height({height}) and width({width}) must be smaller "
                f"than image height({self._image_height}) and width({self._image_width})."
            )
        self._height = height
        self._width = width
        if cy is not None and cx is not None:
            self._check_center_validity(cy, cx)
        self._cy = cy
        self._cx = cx
        # intialize the grid cell center/offset to None
        # they must be set later if the ``Box`` use case is associated with Grid
        self._is_grid_associated = False
        self._cy_offset: Optional[int] = None
        self._cx_offset: Optional[int] = None
        self._grid_cy: Optional[int] = None
        self._grid_cx: Optional[int] = None

    def _check_center_validity(self, cy: Optional[int], cx: Optional[int]):
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
                "Box center cy, cx must be set to check"
                " their numerical validaty."
                "Currently they are None."
            )

        if cy < self._height / 2 or cx < self._width / 2:
            raise ValueError(
                f"Box center({cy}, {cx}) must be greater than half "
                f"height({self._height/2}) and half width({self._width/2})"
                "respectively."
            )

        if (
            cy >= self._image_height - self._height / 2
            or cx >= self._image_width - self._width / 2
        ):
            raise ValueError(
                f"Box center({cy}, {cx}) must be less than half "
                f"height({self._image_height - self._height/2}) and"
                f" half width({self._image_width - self._width/2})"
                "respectively."
            )

    def _check_offset_validity(self, cy_offset: int, cx_offset: int):
        """
        Check the validaty of y coordinate offset and x coordinate offset.

        Args:
            cy_offset: the offset between the box center and the grid cell
                center.
            cx_offset: the offset between the box center and the grid cell
                center.

        Returns:
            True if the offset is valid, False otherwise.
        """
        if self._grid_cy is not None and self._grid_cx is not None:
            # the computed cell center
            computed_cy = self._grid_cy + cy_offset
            computed_cx = self._grid_cx + cx_offset
            if computed_cy < 0 or computed_cx < 0:
                return False

            if (
                computed_cy > self._image_height
                or computed_cx > self._image_width
            ):
                return False

    def set_center(self, cy: int, cx: int):
        """
        Set the y coordinate and x coordinate of the center(a pixel) of the Box.

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
        The row index(unit: pixel) of the grid cell center in the image array.

        Raises:
            ValueError: if the box is not associated with a grid.

        Returns:
            The row index of the grid cell center in the image array.
        """
        if not self.is_grid_associated or self._grid_cy is None:
            raise ValueError(
                "The box is not associated with a grid."
                "Therefore, there is no grid cell center."
            )
        return self._grid_cy

    @property
    def grid_cx(self) -> int:
        """
        The column index(unit: pixel) of the grid cell center in the image
        array.

        Raises:
            ValueError: if the box is not associated with a grid.

        Returns:
            The column index of the grid cell center in the image array.
        """

        if not self.is_grid_associated or self._grid_cx is None:
            raise ValueError(
                "The box is not associated with a grid."
                "Therefore, there is no grid cell center."
            )
        return self._grid_cx

    @property
    def grid_cell_center(self) -> Tuple[int, int]:
        """
        The position of center(a pixel) of the grid cell that the Box is
        associated with.

        Raises:
            ValueError: if the box is not associated with a grid.

        Returns:
            Tuple[int, int]: the center of the grid cell that the Box is associated with.
        """
        if (
            not self.is_grid_associated
            or self._grid_cy is None
            or self._grid_cx is None
        ):
            raise ValueError(
                "The box is not associated with a grid."
                "Therefore, there is no grid cell center."
            )
        return self._grid_cy, self._grid_cx

    def set_offset(
        self, cy_offset: int, cx_offset: int, check_validity: bool = False
    ):
        """
        Set the offset(unit: pixel) of the box center from the grid cell center.

        Args:
            cy_offset: the row index of the box center offset from the grid cell
                center in the image array, the unit is one pixel.
            cx_offset: the column index of the box center offset from the grid
                cell center in the image array, the unit is one pixel.
            check_validity: a boolean indicating whether to check the validity
                of the offset.
        """
        if check_validity:
            self._check_offset_validity(cy_offset, cx_offset)
        self._cy_offset = cy_offset
        self._cx_offset = cx_offset

    @property
    def offset(self):
        """
        The offset(unit: pixel) of the box center from the grid cell center.

        Returns:
            the offset of the box center from the grid cell
        """
        return self._cy_offset, self._cx_offset

    @property
    def cy_offset(self) -> int:
        """
        The row index difference(unit: pixel) between the box center and the
        grid cell.

        Returns:
            The row index difference between the box center and the grid cell.
        """
        if self._cy_offset is not None:
            return self._cy_offset
        if self._grid_cy is not None and self._cy is not None:
            self._cy_offset = self._cy - self._grid_cy
        else:
            self._offset_condition_check()
            raise ValueError("cy_offset is not set.")
        return self._cy_offset

    @property
    def cx_offset(self) -> int:
        """
        The column index difference(unit: pixel) between the box center and the
        grid cell

        Returns:
            The column index difference between the box center and the grid cell
        """
        if self._cx_offset is not None:
            return self._cx_offset
        if self._cx is not None and self._grid_cx is not None:
            self._cx_offset = self._cx - self._grid_cx
        else:
            self._offset_condition_check()
            raise ValueError("cx_offset is not set")
        return self._cx_offset

    @property
    def cy(self) -> int:
        """
        Compute and return row index(unit: pixel) of the box center
        in the image array.
        It returns the row index of the box center in the image array directly
        if the box center is set.
        Otherwise, if it computes and sets the box center y coordinate when the
        box is both associated with a grid and the
        offset is set.

        Returns:
            The row index of the box center in the image array.
        """
        if self._cy is not None:
            return self._cy
        else:
            # if cy computation condition is met, then cy is set
            if self._grid_cy is not None and self._cy_offset is not None:
                self._cy = self._grid_cy + self._cy_offset
            else:
                self._center_condition_check()
                raise ValueError("cy is not set.")
        return self._cy

    @property
    def cx(self) -> int:
        """
        The column index(unit: pixel) of the box center in the image array.
        It returns the column index of the box center in the image array
        directly if the box center is set.
        Otherwise, if it computes and sets the box center x coordinate when the
        box is both associated with a grid and the
        offset is set.

        Returns:
            The column index of the box center in the image array.
        """
        if self._cx is not None:
            return self._cx
        else:
            # if cx computation condition is met, then cx is set
            if self._grid_cx is not None and self._cx_offset is not None:
                self._cx = self._grid_cx + self._cx_offset
            else:
                self._center_condition_check()
                raise ValueError("cx is not set.")
        return self._cx

    @property
    def box_center(self) -> Tuple[int, int]:
        """
        Get the box center y coordinate and x coordinate.
        If box center is not set not computable, it raises a ``ValueError``

        Returns:
            The box center in a ``Tuple`` format.
        """
        return (self.cy, self.cx)

    @property
    def corners(self) -> Tuple[Tuple[int, int], ...]:
        """
        Compute and return the positions of corners of the box, (top left, top right, bottom left, bottom right).

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The corners of the box in a ``Tuple`` format.
        """
        return tuple(
            (self.cy + h_offset, self.cx + w_offset)
            for h_offset in [-self._height // 2, self._height // 2]
            for w_offset in [-self._width // 2, self._width // 2]
        )

    @property
    def box_min_x(self) -> int:
        """
        Compute the minimum x coordinate(unit: pixel) of the box.

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The minimum x coordinate of the box.
        """
        return max(self.cx - round(0.5 * self._width), 0)

    @property
    def box_max_x(self) -> int:
        """
        Compute the maximum x coordinate(unit: pixel) of the box.

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The maximum x coordinate of the box.
        """
        return self.cx + self._width // 2

    @property
    def box_min_y(self) -> int:
        """
        Compute the minimum y coordinate(unit: pixel) of the box.

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The minimum y coordinate of the box.
        """
        return max(self.cy - round(0.5 * self._height), 0)

    @property
    def box_max_y(self) -> int:
        """
        Compute the maximum y coordinate(unit: pixel) of the box.

        Raises:
            ValueError: if the box center is not set.

        Returns:
            The maximum y coordinate of the box.
        """
        return self.cy + round(0.5 * self._height)

    @property
    def area(self) -> int:
        """
        Compute the area of the box(unit: pixel).

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

    def _offset_condition_check(self):
        """
        When the the offset is not set, this function checks the reason the
        offset cannot be computed and raises the corresponding error.

        Raises:
            ValueError: if the grid cell is not associated with the box and
                the box center is not set.
            ValueError: if the grid cell is not associated with the box.
            ValueError: if the box center is not set.
        """
        result_msg = (
            "Hence, the offset of the box center from the grid cell"
            + "center cannot be computed."
        )
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
        When the the center is not set, this function checks the reason the box
        center cannot be computed and raises the corresponding error.

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
        image_payload_idx: the index of the image payload in the DataPack's
            image payload list. If it's not set,
            it defaults to 0 which means it will load the first image payload.
        height: the height of the bounding box, the unit is one pixel.
        width: the width of the bounding box, the unit is one pixel.
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
