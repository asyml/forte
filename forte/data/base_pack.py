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
from typing import List, Optional, Set, Type, TypeVar, Union

import jsonpickle

from forte.common.types import EntryType, LinkType, GroupType
from forte.data.container import EntryContainer
from forte.data.index import BaseIndex
from forte.data.ontology.core import Entry


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

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self.meta, k, v)

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

    def record_entry(self, entry: EntryType):
        r"""

        Record basic information for the entry:
          - Set the id for the entry.
          - Record the creator component for the entry.
          - Record the field creator component for the entry.

        Args:
            entry: The entry to be added.

        Returns:

        """
        # Assign a new id for the entry.
        entry.set_tid()

        # Once we have the id of this entry, we can record the component
        self.add_entry_creation_record(entry.tid)
        for f in entry.get_fields_modified():
            # We record the fields created before pack attachment.
            self.add_field_record(entry.tid, f)
        entry.reset_fields_modified()

    def serialize(self) -> str:
        r"""Serializes a pack to a string."""
        return jsonpickle.encode(self, unpicklable=True)

    @classmethod
    def deserialize(cls, string: str):
        r"""Deserialize a pack from a string.
        """
        return jsonpickle.decode(string)

    def view(self):
        return copy.deepcopy(self)

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
        print(self.creation_records)
        entry_set: Set[int] = self.creation_records[component]

        if len(entry_set) == 0:
            logging.warning("There is no entry generated by '%s' "
                            "in this datapack", component)
        return entry_set

    def get_entries_by_component(self, component: str) -> Set[EntryType]:
        return {self.get_entry(tid)
                for tid in self.get_ids_by_component(component)}

    def get_ids_by_type(self, entry_type: Type[EntryType]) -> Set[int]:
        r"""Look up the type_index with key ``entry_type``.

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

    def get_entries_by_type(self, tp: Type[EntryType]) -> Set[EntryType]:
        entries: Set[EntryType] = set()
        for tid in self.get_ids_by_type(tp):
            entry: EntryType = self.get_entry(tid)
            if isinstance(entry, tp):
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
