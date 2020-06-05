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

from functools import total_ordering
from typing import Optional, Set, Tuple, Type, Any, Dict, Union, Iterable, List

import numpy as np

from forte.common.exception import IncompleteEntryError
from forte.data.base_pack import PackType
from forte.data.ontology.core import Entry, BaseLink, BaseGroup, EntryType, \
    MultiEntry
from forte.data.span import Span

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
]

QueryType = Union[Dict[str, Any], np.ndarray]


class Generics(Entry):
    def __init__(self, pack: PackType):
        super().__init__(pack=pack)


# TODO: Annotation is not hashable.
@total_ordering
class Annotation(Entry):
    r"""Annotation type entries, such as "token", "entity mention" and
    "sentence". Each annotation has a :class:`Span` corresponding to its offset
    in the text.

    Args:
        pack (PackType): The container that this annotation
            will be added to.
        begin (int): The offset of the first character in the annotation.
        end (int): The offset of the last character in the annotation + 1.
    """

    def __init__(self, pack: PackType, begin: int, end: int):
        self._span: Optional[Span]
        self.set_span(begin, end)
        super().__init__(pack)

    @property
    def span(self):
        return self._span

    @property
    def begin(self):
        return self._span.begin

    @property
    def end(self):
        return self._span.end

    def set_span(self, begin: int, end: int):
        r"""Set the span of the annotation.
        """
        self._span = Span(begin, end)

    # Not really good to hash a mutable type.
    # def __hash__(self):
    #     r"""The hash function of :class:`Annotation`.
    #
    #     Users can define their own hash function by themselves but this must
    #     be consistent to :meth:`eq`.
    #     """
    #     return hash(
    #         (type(self), self.pack, self.span.begin, self.span.end)
    #     )

    def __eq__(self, other):
        r"""The eq function of :class:`Annotation`.
        By default, :class:`Annotation` objects are regarded as the same if
        they have the same type, span, and are generated by the same component.

        Users can define their own eq function by themselves but this must
        be consistent to :meth:`hash`.
        """
        if other is None:
            return False
        return (type(self), self.span.begin, self.span.end) == \
               (type(other), other.span.begin, other.span.end)

    def __lt__(self, other):
        r"""To support total_ordering, :class:`Annotations` must provide
        :meth:`__lt__`.
        """
        if self.span != other.span:
            return self.span < other.span
        return (str(type(self)), self._tid) < (str(type(other)), other.tid)

    @property
    def text(self):
        if self.pack is None:
            raise ValueError("Cannot get text because annotation is not "
                             "attached to any data pack.")
        return self.pack.get_span_text(self.span)

    @property
    def index_key(self) -> int:
        return self.tid


class Link(BaseLink):
    r"""Link type entries, such as "predicate link". Each link has a parent node
    and a child node.

    Args:
        pack (EntryContainer): The container that this annotation
            will be added to.
        parent (Entry, optional): the parent entry of the link.
        child (Entry, optional): the child entry of the link.
    """
    # this type Any is needed since subclasses of this class will have new types
    ParentType: Any = Entry
    ChildType: Any = Entry

    def __init__(
            self,
            pack: PackType,
            parent: Optional[Entry] = None,
            child: Optional[Entry] = None
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
                f"instance of {self.ParentType}, but get {type(parent)}")
        self._parent = parent.tid

    def set_child(self, child: Entry):
        r"""This will set the `child` of the current instance with given Entry.
        The child is saved internally by its pack specific index key.

        Args:
            child: The child entry.
        """
        if not isinstance(child, self.ChildType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(child)}")
        self._child = child.tid

    @property
    def parent(self):
        r"""Get ``tid`` of the parent node. To get the object of the parent
        node, call :meth:`get_parent`.
        """
        return self._parent

    @property
    def child(self):
        r"""Get ``tid`` of the child node. To get the object of the child node,
        call :meth:`get_child`.
        """
        return self._child

    def get_parent(self) -> Entry:
        r"""Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the parent of the link.
        """
        if self.pack is None:
            raise ValueError("Cannot get parent because link is not "
                             "attached to any data pack.")
        if self._parent is None:
            raise ValueError("The parent of this entry is not set.")
        return self.pack.get_entry(self._parent)

    def get_child(self) -> Entry:
        r"""Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link.
        """
        if self.pack is None:
            raise ValueError("Cannot get child because link is not"
                             " attached to any data pack.")
        if self._child is None:
            raise ValueError("The child of this entry is not set.")
        return self.pack.get_entry(self._child)


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
            members: Optional[Set[Entry]] = None,
    ):  # pylint: disable=useless-super-delegation
        self._members: Set[int] = set()
        super().__init__(pack, members)

    def add_member(self, member: Entry):
        r"""Add one entry to the group.

        Args:
            member: One member to be added to the group.
        """
        if not isinstance(member, self.MemberType):
            raise TypeError(
                f"The members of {type(self)} should be "
                f"instances of {self.MemberType}, but got {type(member)}")
        self._members.add(member.tid)

    def get_members(self) -> List[Entry]:
        r"""Get the member entries in the group.

        Returns:
             A set of instances of :class:`Entry` that are the members of the
             group.
        """
        if self.pack is None:
            raise ValueError("Cannot get members because group is not "
                             "attached to any data pack.")
        member_entries = []
        for m in self._members:
            member_entries.append(self.pack.get_entry(m))
        return member_entries


class MultiPackGeneric(MultiEntry, Entry):
    def __init__(self, pack: PackType):
        super(MultiPackGeneric, self).__init__(pack=pack)


class MultiPackLink(MultiEntry, BaseLink):
    r"""This is used to link entries in a :class:`MultiPack`, which is
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
        self._parent: Optional[Tuple[int, int]] = None
        self._child: Optional[Tuple[int, int]] = None

        super().__init__(pack, parent, child)

    @property
    def parent(self) -> Tuple[int, int]:
        if self._parent is None:
            raise IncompleteEntryError("Parent is not set for this link.")
        return self._parent

    @property
    def child(self) -> Tuple[int, int]:
        if self._child is None:
            raise IncompleteEntryError("Child is not set for this link.")
        return self._child

    def set_parent(self, parent: Entry):
        r"""This will set the `parent` of the current instance with given Entry.
        The parent is saved internally as a tuple: ``pack index`` and
        ``entry.tid``. Pack index is the index of the data pack in the
        multi-pack.

        Args:
            parent: The parent of the link, which is an Entry from a data pack,
                it has access to the pack index and its own tid in the pack.
        """
        if not isinstance(parent, self.ParentType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(parent)}")
        self._parent = self.pack.get_pack_index(parent.pack_id), parent.tid

    def set_child(self, child: Entry):
        r"""This will set the `child` of the current instance with given Entry.
        The child is saved internally as a tuple: ``pack index`` and
        ``entry.tid``. Pack index is the index of the data pack in the
        multi-pack.

        Args:
            child: The child of the link, which is an Entry from a data pack,
                it has access to the pack index and its own tid in the pack.
        """

        if not isinstance(child, self.ChildType):
            raise TypeError(
                f"The child of {type(self)} should be an "
                f"instance of {self.ChildType}, but get {type(child)}")
        self._child = self.pack.get_pack_index(child.pack_id), child.tid

    def get_parent(self) -> Entry:
        r"""Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the parent of the link.
        """
        if self._parent is None:
            raise IncompleteEntryError("The parent of this link is not set.")

        pack_idx, parent_tid = self._parent
        return self.pack.get_subentry(pack_idx, parent_tid)

    def get_child(self) -> Entry:
        r"""Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link.
        """
        if self._child is None:
            raise IncompleteEntryError("The parent of this link is not set.")

        pack_idx, child_tid = self._child
        return self.pack.get_subentry(pack_idx, child_tid)


# pylint: disable=duplicate-bases
class MultiPackGroup(MultiEntry, BaseGroup[Entry]):
    r"""Group type entries, such as "coreference group". Each group has a set
    of members.
    """
    MemberType: Type[EntryType] = Entry

    def __init__(
            self, pack: PackType, members: Optional[Iterable[Entry]] = None
    ):  # pylint: disable=useless-super-delegation
        self._members: List[Tuple[int, int]] = []
        super().__init__(pack, members)

    def add_member(self, member: Entry):
        if not isinstance(member, self.MemberType):
            raise TypeError(
                f"The members of {type(self)} should be "
                f"instances of {self.MemberType}, but got {type(member)}")

        self._members.append(
            (self.pack.get_pack_index(member.pack_id), member.tid))

    def get_members(self) -> List[Entry]:
        members = []
        for pack_idx, member_tid in self._members:
            members.append(self.pack.get_subentry(pack_idx, member_tid))
        return members


class Query(Generics):
    r"""An entry type representing queries for information retrieval tasks.

    Args:
        pack (Data pack): Data pack reference to which this query will be added
    """

    def __init__(self, pack: PackType):
        super().__init__(pack)
        self._value: Optional[QueryType] = None
        self._results: Dict[str, float] = {}

    @property
    def value(self) -> QueryType:
        return self._value

    @value.setter
    def value(self, value: QueryType):
        r"""Sets the value of the query.

        Args:
            value (numpy array or str): A vector or a string (in case of
            traditional models) representing the query.
        """
        self._value = value

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, pid_to_score: Dict[str, float]):
        self._results = pid_to_score

    def add_result(self, pid: str, score: float):
        """
        Set the result score for a particular pack (based on the pack id).

        Args:
            pid: the pack id.
            score: the score for the pack

        Returns:

        """
        self.results[pid] = score

    def update_results(self, pid_to_score: Dict[str, float]):
        r"""Updates the results for this query.

        Args:
             pid_to_score (dict): A dict containing pack id -> score mapping
        """
        self.results.update(pid_to_score)


SinglePackEntries = (Link, Group, Annotation, Generics)
MultiPackEntries = (MultiPackLink, MultiPackGroup, MultiPackGeneric)
