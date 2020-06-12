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
from abc import abstractmethod
from typing import List, Optional, Set, Type, TypeVar, Union, Iterator, Dict, \
    Tuple, Any

import jsonpickle

from forte.common import ProcessExecutionException, EntryNotFoundError
from forte.data.container import EntryContainer
from forte.data.index import BaseIndex
from forte.data.ontology.core import (Entry, EntryType, GroupType, LinkType)

__all__ = [
    "BasePack",
    "BaseMeta",
    "PackType"
]

from forte.pack_manager import PackManager


class BaseMeta:
    r"""Basic Meta information for both :class:`~forte.data.data_pack.DataPack`
    and :class:`~forte.data.multi_pack.MultiPack`.

    Args:
        pack_name:  An name to identify the data pack, which is helpful in
           situation like serialization. It is suggested that the packs should
           have different doc ids.

    """

    def __init__(self, pack_name: Optional[str] = None):
        self.pack_name: Optional[str] = pack_name
        self._pack_id: int = -1

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Re-obtain the pack manager during deserialization.
        Args:
            state:

        Returns:

        """
        self.__dict__.update(state)

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
        pack_manager( PackManager): The pack manager to manage the ids of the
            packs.
        pack_name (str, optional): a string name of the pack.

    """

    # pylint: disable=too-many-public-methods
    def __init__(self, pack_manager: PackManager,
                 pack_name: Optional[str] = None):
        super().__init__()
        self._pack_manager = pack_manager

        self.links: List[LinkType] = []
        self.groups: List[GroupType] = []

        self.meta: BaseMeta = self._init_meta(pack_name)
        self.index: BaseIndex = BaseIndex()

        # Assign a pack id for this pack.
        self._pack_manager.set_pack_id(self)
        self.__control_component: Optional[str] = None
        self._pending_entries: Dict[int, Tuple[Entry, Optional[str]]] = {}

    def __getstate__(self):
        state = super().__getstate__()
        state.pop('index')
        state.pop('_pending_entries')
        state.pop('_BasePack__control_component')
        state.pop('_pack_manager')

        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__['_pending_entries'] = {}
        self.__control_component: Optional[str] = None

    @abstractmethod
    def _init_meta(self, pack_name: Optional[str] = None) -> BaseMeta:
        raise NotImplementedError

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self.meta, k, v)

    @property
    def pack_id(self):
        return self.meta.pack_id

    @pack_id.setter
    def pack_id(self, pack_id: int):
        self.meta.pack_id = pack_id

    @abstractmethod
    def __iter__(self) -> Iterator[EntryType]:
        raise NotImplementedError

    def __del__(self):
        if len(self._pending_entries) > 0:
            raise ProcessExecutionException(
                f"There are {len(self._pending_entries)} "
                f"entries not added to the index correctly.")

    @property
    def pack_name(self):
        return self.meta.pack_name

    @pack_name.setter
    def pack_name(self, pack_name: str):
        """
        Update the pack name of this pack.

        Args:
            pack_name: The new doc id.

        Returns:

        """
        self.meta.pack_name = pack_name

    @abstractmethod
    def delete_entry(self, entry: EntryType):
        r""" Remove the entry from the pack.

        Args:
            entry: The entry to be removed.

        Returns:

        """
        raise NotImplementedError

    def add_entry(self, entry: Entry,
                  component_name: Optional[str] = None) -> EntryType:
        r"""Add an :class:`~forte.data.ontology.top.Entry` object to the
        :class:`BasePack` object. Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be added to the pack.
            component_name (str): A name to record that the entry is created by
             this component.

        Returns:
            The input entry itself
        """
        # When added to the pack, make a record.
        self.record_entry(entry, component_name)
        # TODO: Returning the entry itself may not be helpful.
        return self._add_entry(entry)

    @abstractmethod
    def _add_entry(self, entry: Entry) -> EntryType:
        r"""Add an :class:`~forte.data.ontology.top.Entry` object to the
        :class:`BasePack` object. Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be added to the pack.

        Returns:
            The input entry itself
        """
        raise NotImplementedError

    def add_all_remaining_entries(self, component: Optional[str] = None):
        """
        Calling this function will add the entries that are not added to the
        pack manually.

        Args:
            component (str): Overwrite the component record with this.

        Returns:

        """
        for entry, c in list(self._pending_entries.values()):
            c_ = component if component else c
            self.add_entry(entry, c_)
        self._pending_entries.clear()

    def serialize(self, drop_record: Optional[bool] = False) -> str:
        r"""Serializes a pack to a string."""
        if drop_record:
            self.creation_records.clear()
            self.field_records.clear()

        return jsonpickle.encode(self, unpicklable=True)

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

    def record_entry(self, entry: Entry, component_name: Optional[str] = None):
        c = component_name

        if c is None:
            # Use the auto-inferred control component.
            c = self.__control_component

        if c is not None:
            try:
                self.creation_records[c].add(entry.tid)
            except KeyError:
                self.creation_records[c] = {entry.tid}

    def record_field(self, entry_id: int, field_name: str):
        """
        Record who modifies the entry, will be called
        in :class:`~forte.data.ontology.core.Entry`

        Args:
            entry_id: The id of the entry.
            field_name: The name of the field modified.

        Returns:

        """
        c = self.__control_component

        if c is not None:
            try:
                self.field_records[c].add((entry_id, field_name))
            except KeyError:
                self.field_records[c] = {(entry_id, field_name)}

    def on_entry_creation(self, entry: Entry,
                          component_name: Optional[str] = None):
        """
        Call this when adding a new entry, will be called
        in :class:`~forte.data.ontology.core.Entry` when
        its `__init__` function is called.

        Args:
            entry (Entry): The entry to be added.
            component_name (str): A name to record that the entry is created by
             this component.

        Returns:

        """
        c = component_name

        if c is None:
            # Use the auto-inferred control component.
            c = self.__control_component

        # Record that this entry hasn't been added to the index yet.
        self._pending_entries[entry.tid] = entry, c

    def regret_creation(self, entry: EntryType):
        """

        Args:
            entry:

        Returns:

        """
        self._pending_entries.pop(entry.tid)

    # TODO: how to make this return the precise type here?
    def get_entry(self, tid: int) -> EntryType:
        r"""Look up the entry_index with key ``ptr``. Specific implementation
        depends on the actual class."""
        entry: EntryType = self.index.get_entry(tid)
        if entry is None:
            raise KeyError(
                f"There is no entry with tid '{tid}'' in this datapack")
        return entry

    @abstractmethod
    def get_data(
            self, context_type, request, skip_k
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get(self, entry_type: Type[EntryType], **kwargs):
        raise NotImplementedError

    def get_single(self, entry_type: Type[EntryType]) -> EntryType:
        r"""Take a single entry of type :attr:`entry_type` from this data
        pack. This is useful when the target entry type appears only one
        time in the :class:`DataPack` for e.g., a Document entry. Or you just
        intended to take the first one.

        Args:
            entry_type: The entry type to be retrieved.

        Returns:
            A single data entry.
        """
        for a in self.get(entry_type):
            return a

        raise EntryNotFoundError(
            f"The entry {entry_type} is not found in the provided pack.")

    def get_ids_by_component(self, component: str) -> Set[int]:
        r"""Look up the component_index with key ``component``."""
        entry_set: Set[int] = self.creation_records[component]
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

    def get_ids_by_components(self, components: List[str]) -> Set[int]:
        """Look up component_index using a list of components."""
        valid_component_id: Set[int] = set()
        for component in components:
            valid_component_id |= self.get_ids_by_component(component)
        return valid_component_id

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
        return subclass_index

    def get_entries_by_type(
            self, entry_type: Type[EntryType]) -> List[EntryType]:
        """
        Return all entries of this particular type without orders. If you
        need to use natural order of the annotations, use
        :func:`forte.data.data_pack.get_entries`.

        Args:
            entry_type: The type of the entry you are looking for.

        Returns:

        """
        entries: List[EntryType] = []
        for tid in self.get_ids_by_type(entry_type):
            entry: EntryType = self.get_entry(tid)
            if isinstance(entry, entry_type):
                entries.append(entry)
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
    ) -> List[LinkType]:
        links: List[LinkType] = []
        if isinstance(node, Entry):
            tid = node.tid
            if tid is None:
                raise ValueError("The requested node has no tid. "
                                 "Have you add this entry into the datapack?")
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
                links.append(entry)  # type: ignore
        return links

    def get_links_by_parent(
            self, parent: Union[int, EntryType]) -> List[LinkType]:
        return self.get_links_from_node(parent, True)

    def get_links_by_child(
            self, child: Union[int, EntryType]) -> List[LinkType]:
        return self.get_links_from_node(child, False)

    def get_groups_by_member(
            self, member: Union[int, EntryType]) -> Set[GroupType]:
        groups: Set[GroupType] = set()
        if isinstance(member, Entry):
            tid = member.tid
            if tid is None:
                raise ValueError("Argument member has no tid. "
                                 "Have you add this entry into the datapack?")
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
