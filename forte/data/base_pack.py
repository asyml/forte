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
import uuid
from abc import abstractmethod
from typing import (
    List, Optional, Set, Type, TypeVar, Union, Iterator, Dict, Tuple, Any,
    Iterable)

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


class BaseMeta:
    r"""Basic Meta information for both :class:`~forte.data.data_pack.DataPack`
    and :class:`~forte.data.multi_pack.MultiPack`.

    Args:
        pack_name:  An name to identify the data pack, which is helpful in
           situation like serialization. It is suggested that the packs should
           have different doc ids.
    Attributes:
        record: Initialized as a dictionary. This is not a required field.
            The key of the record should be the entry type and values should
            be attributes of the entry type. All the information would be used
            for consistency checking purpose if
            :meth:`~forte.pipeline.Pipeline.enforce_consistency` is enabled
            for the pipeline.
    """

    def __init__(self, pack_name: Optional[str] = None):
        self.pack_name: Optional[str] = pack_name
        self._pack_id: int = uuid.uuid4().int
        self.record: Dict[str, Set[str]] = dict()

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


class BasePack(EntryContainer[EntryType, LinkType, GroupType]):
    r"""The base class of :class:`~forte.data.data_pack.DataPack` and
    :class:`~forte.data.multi_pack.MultiPack`.

    Args:
        pack_name (str, optional): a string name of the pack.

    """

    # pylint: disable=too-many-public-methods
    def __init__(self, pack_name: Optional[str] = None):
        super().__init__()
        self.links: List[LinkType] = []
        self.groups: List[GroupType] = []

        self._meta: BaseMeta = self._init_meta(pack_name)
        self._index: BaseIndex = BaseIndex()

        self.__control_component: Optional[str] = None

        # This Dict maintains a mapping from entry's tid to the Entry object
        # itself and the component name associated with the entry.
        # The component name is used for tracking the "creator" of this entry.
        self._pending_entries: Dict[int, Tuple[Entry, Optional[str]]] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_index')
        state.pop('_pending_entries')
        state.pop('_BasePack__control_component')
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if 'meta' in self.__dict__:
            self._meta = self.__dict__.pop('meta')
        self.__control_component = None
        self._pending_entries = {}

    @abstractmethod
    def _init_meta(self, pack_name: Optional[str] = None) -> BaseMeta:
        raise NotImplementedError

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self._meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self._meta, k, v)

    @property
    def pack_id(self):
        return self._meta.pack_id

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
        return self._meta.pack_name

    @pack_name.setter
    def pack_name(self, pack_name: str):
        """
        Update the pack name of this pack.

        Args:
            pack_name: The new doc id.

        Returns:

        """
        self._meta.pack_name = pack_name

    @classmethod
    def _deserialize(cls, string: str) -> "PackType":
        """
        This function should deserialize a Pack from a string. The
         implementation should decide the specific pack type.

        Args:
            string: The serialized string to be deserialized.

        Returns:
            An pack object deserialized from the string.
        """
        pack = jsonpickle.decode(string)
        return pack

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
        r"""Add an :class:`~forte.data.ontology.core.Entry` object to the
        :class:`BasePack` object. Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.core.Entry`
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
        r"""Add an :class:`~forte.data.ontology.core.Entry` object to the
        :class:`BasePack` object. Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.core.Entry`
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
            self._creation_records.clear()
            self._field_records.clear()

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
                self._creation_records[c].add(entry.tid)
            except KeyError:
                self._creation_records[c] = {entry.tid}

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
                self._field_records[c].add((entry_id, field_name))
            except KeyError:
                self._field_records[c] = {(entry_id, field_name)}

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
        Will remove the entry from the pending entries internal state of the
        pack.

        Args:
            entry: The entry that we would not add the the pack anymore.

        Returns:

        """
        self._pending_entries.pop(entry.tid)

    # TODO: how to make this return the precise type here?
    def get_entry(self, tid: int) -> EntryType:
        r"""Look up the entry_index with key ``ptr``. Specific implementation
        depends on the actual class."""
        entry: EntryType = self._index.get_entry(tid)
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
    def get(self, entry_type: Type[EntryType], **kwargs) -> Iterator[EntryType]:
        """
        Implementation of this method should provide to obtain the entries in
        entry ordering. If there are orders defined between the entries, they
        should be used first. Otherwise, the insertion order should be
        used (FIFO).

        Args:
            entry_type: The type of the entry to obtain.

        Returns:
            An iterator of the entries matching the provided arguments.
        """
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

    def get_ids_by_creator(self, component: str) -> Set[int]:
        r"""
        Look up the component_index with key `component`. This will return
        the entry ids that are created by the `component`

        Args:
            component: The component (creator) to find ids for.

        Returns:
            A set of entry ids that are created by the component.
        """
        entry_set: Set[int] = self._creation_records[component]
        return entry_set

    def is_created_by(
            self, entry: Entry,
            components: Union[str, Iterable[str]]) -> bool:
        """
        Check if the entry is created by any of the provided components.

        Args:
            entry: The entry to check.
            components: The list of component names.

        Returns (bool):
            True if the entry is created by the component, False otherwise.
        """
        if isinstance(components, str):
            components = [components]

        for c in components:
            if entry.tid in self._creation_records[c]:
                break
        else:
            # The entry not created by any of these components.
            return False

        return True

    def get_entries_from(self, component: str) -> Set[EntryType]:
        """
        Look up all entries from the `component` as a unordered set

        Args:
            component: The component (creator) to get the entries. It is
            normally the full qualified name of the creator class, but it
            may also be customized based on the implementation.

        Returns:
            The set of entry ids that are created by the input component.
        """
        return {self.get_entry(tid)
                for tid in self.get_ids_by_creator(component)}

    def get_ids_from(self, components: List[str]) -> Set[int]:
        """
        Look up entries using a list of components (creators). This will find
        each creator iteratively and combine the result.

        Args:
            components (List[str]): The list of components to find.

        Returns:
            The list of entry ids that are created from these components.
        """
        valid_component_id: Set[int] = set()
        for component in components:
            valid_component_id |= self.get_ids_by_creator(component)
        return valid_component_id

    def get_ids_by_type_subtype(
            self, entry_type: Type[EntryType]) -> Set[int]:
        r"""Look up the type_index with key ``entry_type``.

        Args:
            entry_type: The type of the entry you are looking for.

        Returns:
             A set of entry ids. The entries are instances of `entry_type` (
             and also includes instances of the subclasses of `entry_type`).
        """
        subclass_index: Set[int] = set()
        for index_key, index_val in self._index.iter_type_index():
            if issubclass(index_key, entry_type):
                subclass_index.update(index_val)
        return subclass_index

    def _expand_to_sub_types(self, entry_type: Type[EntryType]) -> Set[Type]:
        """
        Return all the types and the sub types that inherit from the provided
        type.

        Args:
            entry_type: The provided type to search for entry.

        Returns:
            A set of all the sub-types extending the provided type, including
            the input `entry_type` itself.
        """
        all_types: Set[Type] = set()
        for data_type in self._index.indexed_types():
            if issubclass(data_type, entry_type):
                all_types.add(data_type)
        return all_types

    def get_entries_of(
            self, entry_type: Type[EntryType], exclude_sub_types=False
    ) -> Iterator[EntryType]:
        """
        Return all entries of this particular type without orders. If you
        need to get the annotations based on the entry ordering,
        use :meth:`forte.data.base_pack.get`.

        Args:
            entry_type: The type of the entry you are looking for.
            exclude_sub_types (bool): Whether to ignore the inherited sub type
            of the provided `entry_type`. Default is True.

        Returns:
            An iterator of the entries matching the type constraint.
        """
        if exclude_sub_types:
            for tid in self._index.query_by_type(entry_type):
                yield self.get_entry(tid)
        else:
            for tid in self.get_ids_by_type_subtype(entry_type):
                yield self.get_entry(tid)

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

        if not self._index.link_index_on:
            self._index.build_link_index(self.links)

        for tid in self._index.link_index(tid, as_parent=as_parent):
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

        if not self._index.group_index_on:
            self._index.build_group_index(self.groups)

        for tid in self._index.group_index(tid):
            entry: EntryType = self.get_entry(tid)
            if self.validate_group(entry):
                groups.add(entry)  # type: ignore
        return groups


PackType = TypeVar('PackType', bound=BasePack)
