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

import copy
import logging
from abc import abstractmethod
from typing import List, Optional, Set, Type, TypeVar, Union, Iterator

import jsonpickle

from forte.data.container import EntryContainer
from forte.data.index import BaseIndex
from forte.data.ontology.core import Entry
from forte.data.ontology.core import EntryType, GroupType, LinkType
from forte.pack_manager import PackManager

__all__ = [
    "BasePack",
    "BaseMeta",
    "PackType"
]


class BaseMeta:
    r"""Basic Meta information for both :class:`~forte.data.data_pack.DataPack`
    and :class:`~forte.data.multi_pack.MultiPack`.
    """

    def __init__(self, doc_id: Optional[str] = None):
        self.doc_id: Optional[str] = doc_id
        self._pack_id: int = -1
        # Obtain the global pack manager.
        self._pack_manager: PackManager = PackManager()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_pack_manager')
        return state

    def __setstate__(self, state):
        """
        Re-obtain the pack manager during deserialization.
        Args:
            state:

        Returns:

        """
        self.__dict__.update(state)
        self._pack_manager: PackManager = PackManager()

    @property
    def pack_id(self) -> int:
        return self._pack_id

    @pack_id.setter
    def pack_id(self, pid: int):
        self._pack_id = pid


class BasePack(EntryContainer[EntryType, LinkType, GroupType]):
    r"""The base class of :class:`~forte.data.data_pack.DataPack` and
    :class:`~forte.data.multi_pack.MultiPack`.

    Args:
        doc_id (str, optional): a string identifier of the pack.

    """

    # pylint: disable=too-many-public-methods
    def __init__(self, doc_id: Optional[str] = None):
        super().__init__()

        self.links: List[LinkType] = []
        self.groups: List[GroupType] = []

        self.meta: BaseMeta = BaseMeta(doc_id)
        self.index: BaseIndex = BaseIndex()

        # Obtain the global pack manager.
        self._pack_manager: PackManager = PackManager()

        self.__control_component: Optional[str] = None

    def __getstate__(self):
        state = super().__getstate__()
        state.pop('index')
        state.pop('_pack_manager')
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__['_pack_manager'] = PackManager()

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self.meta, k, v)

    @abstractmethod
    def __iter__(self) -> Iterator[EntryType]:
        raise NotImplementedError

    @abstractmethod
    def delete_entry(self, entry: EntryType):
        r""" Remove the entry from the pack.

        Args:
            entry: The entry to be removed.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def add_entry(self, entry: EntryType) -> EntryType:
        r"""Force add an :class:`~forte.data.ontology.top.Entry` object to the
        :class:`BasePack` object. Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be added to the pack.

        Returns:
            The input entry itself
        """
        raise NotImplementedError

    def add_entry_(self, entry: EntryType) -> int:
        """
        A slightly different variation from `add_entry` function, it returns
        the entry id instead.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be added to the pack.

        Returns:
            The entry id of the added entry.
        """
        return self.add_entry(entry).tid

    @abstractmethod
    def add_or_get_entry(self, entry: EntryType) -> EntryType:
        r"""Try to add an :class:`~forte.data.ontology.top.Entry` object to the
        :class:`BasePack` object.

        If a same entry already exists, will return the existing entry
        instead of adding the new one. Note that we regard two entries as the
        same if their :meth:`~forte.data.ontology.top.Entry.eq` have
        the same return value, and users could
        override :meth:`~forte.data.ontology.top.Entry.eq` in their
        custom entry classes.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be added to the pack.

        Returns:
            If a same entry already exists, returns the existing
            entry. Otherwise, return the (input) entry just added.
        """
        raise NotImplementedError

    def serialize(self) -> str:
        r"""Serializes a pack to a string."""
        return jsonpickle.encode(self, unpicklable=True)

    @staticmethod
    def deserialize(string: str):
        r"""Deserialize a pack from a string.
        """
        return jsonpickle.decode(string)

    def view(self):
        return copy.deepcopy(self)

    def set_control_component(self, component: str):
        """
        Record the current component that is taking control of this pack.

        Args:
            component: The component that is going to take control

        Returns:

        """
        self.__control_component = component

    def add_entry_creation_record(self, entry_id: int):
        """
        Record who creates the entry, will be called
        in :class:`~forte.data.ontology.core.Entry`

        Args:
            entry_id: The id of the entry.

        Returns:

        """
        c = self.__control_component

        if c is None:
            c = self._pack_manager.get_input_source()

        try:
            self.creation_records[c].add(entry_id)
        except KeyError:
            self.creation_records[c] = {entry_id}

    def add_field_record(self, entry_id: int, field_name: str):
        """
        Record who modifies the entry, will be called
        in :class:`~forte.data.ontology.core.Entry`

        Args:
            entry_id: The id of the entry.
            field_name: The name of the field modified.

        Returns:

        """
        c = self.__control_component

        if c is None:
            c = self._pack_manager.get_input_source()

        try:
            self.field_records[c].add((entry_id, field_name))
        except KeyError:
            self.field_records[c] = {(entry_id, field_name)}

    # TODO: how to make this return the precise type here?
    def get_entry(self, tid: int) -> EntryType:
        r"""Look up the entry_index with key ``tid``."""
        entry: EntryType = self.index.get_entry(tid)
        if entry is None:
            raise KeyError(
                f"There is no entry with tid '{tid}'' in this datapack")
        return entry

    def get_ids_by_component(self, component: str) -> Set[int]:
        r"""Look up the component_index with key ``component``."""
        entry_set: Set[int] = self.creation_records[component]

        if len(entry_set) == 0:
            logging.warning("There is no entry generated by '%s' "
                            "in this datapack", component)
        return entry_set

    def get_entries_by_component(self, component: str) -> Set[EntryType]:
        """
        Return all entries created by the particular component, an unordered
        set.

        Args:
            component: The component to get the entries.

        Returns:

        """
        return {self.get_entry(tid)
                for tid in self.get_ids_by_component(component)}

    def get_ids_by_type(self, entry_type: Type[EntryType]) -> Set[int]:
        r"""Look up the type_index with key ``entry_type``.

        Args:
            entry_type: The type of the entry you are looking for.

        Returns:
             A set of entry tids. The entries are instances of entry_type (
             and also includes instances of the subclasses of entry_type).
        """
        subclass_index: Set[int] = set()
        for index_key, index_val in self.index.iter_type_index():
            if issubclass(index_key, entry_type):
                subclass_index.update(index_val)

        if len(subclass_index) == 0:
            logging.warning(
                "There is no %s type entry in this datapack", entry_type)
        return subclass_index

    def get_entries_by_type(
            self, entry_type: Type[EntryType]) -> Set[EntryType]:
        """
        Return all entries of this particular type without orders. If you
        need to use natural order of the annotations, use
        :func:`forte.data.data_pack.get_entries`.

        Args:
            entry_type: The type of the entry you are looking for.

        Returns:

        """
        entries: Set[EntryType] = set()
        for tid in self.get_ids_by_type(entry_type):
            entry: EntryType = self.get_entry(tid)
            if isinstance(entry, entry_type):
                entries.add(entry)
        return entries

    @classmethod
    @abstractmethod
    def validate_link(cls, entry: EntryType) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def validate_group(cls, entry: EntryType) -> bool:
        raise NotImplementedError

    def get_links_from_node(
            self,
            node: Union[int, EntryType],
            as_parent: bool
    ) -> Set[LinkType]:
        links: Set[LinkType] = set()
        if isinstance(node, Entry):
            tid = node.tid
            if tid is None:
                raise ValueError(f"The requested node has no tid. "
                                 f"Have you add this entry into the datapack?")
        elif isinstance(node, int):
            tid = node
        else:
            raise TypeError("Can only get group via entry id (int) or the "
                            "group object itself (Entry).")

        if not self.index.link_index_on:
            self.index.build_link_index(self.links)

        for tid in self.index.link_index(tid, as_parent=as_parent):
            entry: EntryType = self.get_entry(tid)
            if self.validate_link(entry):
                links.add(entry)  # type: ignore
        return links

    def get_links_by_parent(
            self, parent: Union[int, EntryType]) -> Set[LinkType]:
        return self.get_links_from_node(parent, True)

    def get_links_by_child(self, child: Union[int, EntryType]) -> Set[LinkType]:
        return self.get_links_from_node(child, False)

    def get_groups_by_member(
            self, member: Union[int, EntryType]) -> Set[GroupType]:
        groups: Set[GroupType] = set()
        if isinstance(member, Entry):
            tid = member.tid
            if tid is None:
                raise ValueError(f"Argument member has no tid. "
                                 f"Have you add this entry into the datapack?")
        elif isinstance(member, int):
            tid = member
        else:
            raise TypeError("Can only get group via entry id (int) or the "
                            "group object itself (Entry).")

        if not self.index.group_index_on:
            self.index.build_group_index(self.groups)

        for tid in self.index.group_index(tid):
            entry: EntryType = self.get_entry(tid)
            if self.validate_group(entry):
                groups.add(entry)  # type: ignore
        return groups


PackType = TypeVar('PackType', bound=BasePack)
