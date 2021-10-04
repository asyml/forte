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
from typing import Optional, Set, Tuple, Type, Any, Dict, Union, Iterable, List

import numpy as np

from forte.data.base_pack import PackType
from forte.data.ontology.core import (
    Entry,
    BaseLink,
    BaseGroup,
    MultiEntry,
    EntryType,
)
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
        self._span: Optional[Span] = None
        self._begin: int = begin
        self._end: int = end
        super().__init__(pack)

    def __getstate__(self):
        r"""For serializing Annotation, we should create Span annotations for
        compatibility purposes.
        """
        self._span = Span(self._begin, self._end)
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
            self._span = Span(self._begin, self._end)
        return self._span

    @property
    def begin(self):
        return self._begin

    @property
    def end(self):
        return self._end

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
        This function wraps the :meth:`~forte.data.DataPack.get()` method to find
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

            In the above code snippet, we get entries of type ``Token`` within
            each ``sentence`` which were generated by ``NLTKTokenizer``. You
            can consider build coverage index between `Token` and `Sentence`
            if this snippet is frequently used.

        Args:
            entry_type (type): The type of entries requested.
            components (str or list, optional): The component (creator)
                generating the entries requested. If `None`, will return valid
                entries generated by any component.
            include_sub_type (bool): whether to consider the sub types of
                the provided entry type. Default `True`.

        Yields:
            Each `Entry` found using this method.

        """
        yield from self.pack.get(entry_type, self, components, include_sub_type)


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
        import pdb
        pdb.set_trace()
        if not isinstance(parent, self.ParentType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(parent)}"
            )
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
                f"instance of {self.ParentType}, but get {type(child)}"
            )
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
            raise ValueError(
                "Cannot get parent because link is not "
                "attached to any data pack."
            )
        if self._parent is None:
            raise ValueError("The parent of this entry is not set.")
        return self.pack.get_entry(self._parent)

    def get_child(self) -> Entry:
        r"""Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link.
        """
        if self.pack is None:
            raise ValueError(
                "Cannot get child because link is not"
                " attached to any data pack."
            )
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
        members: Optional[Iterable[Entry]] = None,
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
                f"instances of {self.MemberType}, but got {type(member)}"
            )
        self._members.add(member.tid)

    def get_members(self) -> List[Entry]:
        r"""Get the member entries in the group.

        Returns:
             A set of instances of :class:`Entry` that are the members of the
             group.
        """
        if self.pack is None:
            raise ValueError(
                "Cannot get members because group is not "
                "attached to any data pack."
            )
        member_entries = []
        for m in self._members:
            member_entries.append(self.pack.get_entry(m))
        return member_entries


class MultiPackGeneric(MultiEntry, Entry):
    def __init__(self, pack: PackType):
        super().__init__(pack=pack)


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

        super().__init__(pack)

        if parent is not None:
            self.set_parent(parent)
        if child is not None:
            self.set_child(child)

    @property
    def parent(self) -> Tuple[int, int]:
        if self._parent is None:
            raise ValueError("Parent is not set for this link.")
        return self._parent

    @property
    def child(self) -> Tuple[int, int]:
        if self._child is None:
            raise ValueError("Child is not set for this link.")
        return self._child

    def parent_id(self) -> int:
        """
        Return the `tid` of the parent entry.

        Returns: The `tid` of the parent entry.
        """
        return self.parent[1]

    def child_id(self) -> int:
        """
        Return the `tid` of the child entry.

        Returns: The `tid` of the child entry.
        """
        return self.child[1]

    def parent_pack_id(self) -> int:
        """
        Return the `pack_id` of the parent pack.

        Returns: The `pack_id` of the parent pack..
        """
        if self._parent is None:
            raise ValueError("Parent is not set for this link.")
        return self.pack.packs[self._parent[0]].pack_id

    def child_pack_id(self) -> int:
        """
        Return the `pack_id` of the child pack.

        Returns: The `pack_id` of the child pack.
        """
        if self._child is None:
            raise ValueError("Child is not set for this link.")
        return self.pack.packs[self._child[0]].pack_id

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
                f"instance of {self.ParentType}, but get {type(parent)}"
            )
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
                f"instance of {self.ChildType}, but get {type(child)}"
            )
        self._child = self.pack.get_pack_index(child.pack_id), child.tid

    def get_parent(self) -> Entry:
        r"""Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the parent of the link.
        """
        if self._parent is None:
            raise ValueError("The parent of this link is not set.")

        pack_idx, parent_tid = self._parent
        return self.pack.get_subentry(pack_idx, parent_tid)

    def get_child(self) -> Entry:
        r"""Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link.
        """
        if self._child is None:
            raise ValueError("The parent of this link is not set.")

        pack_idx, child_tid = self._child
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
        self._members: List[Tuple[int, int]] = []
        super().__init__(pack)
        if members is not None:
            self.add_members(members)

    def add_member(self, member: Entry):
        if not isinstance(member, self.MemberType):
            raise TypeError(
                f"The members of {type(self)} should be "
                f"instances of {self.MemberType}, but got {type(member)}"
            )

        self._members.append(
            (self.pack.get_pack_index(member.pack_id), member.tid)
        )

    def get_members(self) -> List[Entry]:
        members = []
        for pack_idx, member_tid in self._members:
            members.append(self.pack.get_subentry(pack_idx, member_tid))
        return members


@dataclass
class Query(Generics):
    r"""An entry type representing queries for information retrieval tasks.

    Args:
        pack (Data pack): Data pack reference to which this query will be added
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
