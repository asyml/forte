import copy
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import (Dict, List, Optional, Set, Type, Tuple, TypeVar, Union)

import jsonpickle

from forte.common.types import EntryType
from forte.data.container import EntryContainer
from forte.data.index import BaseIndex
from forte.data.ontology.core import LinkType, GroupType, Entry

logger = logging.getLogger(__name__)

__all__ = [
    "BasePack",
    "BaseMeta",
    "InternalMeta",
    "PackType"
]


class BaseMeta:
    """
    Basic Meta information for both
    :class:`~forte.data.data_pack.DataPack` and
    :class:`~forte.data.multi_pack.MultiPack`.
    """

    def __init__(self, doc_id: Optional[str] = None):
        self.doc_id: Optional[str] = doc_id

        # TODO: These two are definitely internal.
        # the pack has been processed by which processor in the pipeline
        self.process_state: str = ''
        # the pack has been chached by which processor in the pipeline
        self.cache_state: str = ''


class InternalMeta:
    """
    The internal meta information of **one kind of entry** in a datapack.
    Record the entry fields created in the :class:`BasePack` and the entry
    counters.

    Note that the :attr:`internal_metas` in :class:`BasePack` is a dict in
    which the keys are entries types and the values are objects of
    :class:`InternalMeta`.
    """

    def __init__(self):
        self.id_counter = 0
        self.fields_created = defaultdict(set)
        self.default_component = None

        # TODO: Finish the update of this true component_records.
        # A index of the component records of entries and fields. These will
        # indicate "who" created the entry and modified the fields.
        self.component_records: Dict[
            str,  # The component name.
            Set[int],  # The set of entries created by this component.
            Set[  # The set of fields created by this component.
                Tuple[int, str]  # The 2-tuple identify the entry field.
            ]
        ]


class BasePack(EntryContainer[EntryType, LinkType, GroupType]):
    """
    The base class of :class:`~forte.data.data_pack.DataPack` and
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
        self.internal_metas: \
            Dict[type, InternalMeta] = defaultdict(InternalMeta)

        # This is used internally when a processor takes the ownership of this
        # DataPack.
        self._owner_component: str = '__default__'
        self.__poison: bool = False

    def enter_processing(self, component_name: str):
        self._owner_component = component_name

    def current_component(self):
        return self._owner_component

    def exit_processing(self):
        self._owner_component = '__default__'

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self.meta, k, v)

    def set_as_poison(self):
        self.__poison = True

    def is_poison(self) -> bool:
        """
        Indicate that this is a poison (a placeholder element that indicate
        the end of the flow).
        Returns:

        """
        return self.__poison

    @abstractmethod
    def validate(self, entry: EntryType) -> bool:
        """
        Validate whether this type can be added.

        Args:
            entry:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def add_entry(self, entry: EntryType) -> EntryType:
        """
        Force add an :class:`~forte.data.ontology.top.Entry` object to
        the :class:`BasePack` object.
        Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be added to the pack.

        Returns:
            The input entry itself
        """
        raise NotImplementedError

    @abstractmethod
    def add_or_get_entry(self, entry: EntryType) -> EntryType:
        """
        Try to add an :class:`~forte.data.ontology.top.Entry` object to
        the :class:`BasePack` object.
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

    @abstractmethod
    def record_fields(self, fields: List[str], entry_type: Type[EntryType],
                      component: str):
        """Record in the internal meta that the ``entry_type`` entires generated
        by ``component`` have ``fields``.

        If ``component`` is "_ALL_", we will record the ``fields`` for all
        entries of the type ``entry_type`` regardless of their component in
        the internal meta.
        """
        raise NotImplementedError

    def serialize(self) -> str:
        """
        Serializes a pack to a string.
        """
        return jsonpickle.encode(self, unpicklable=True)

    def view(self):
        return copy.deepcopy(self)

    # TODO: how to make this return the precise type here?
    def get_entry(self, tid: str) -> EntryType:
        """
        Look up the entry_index with key ``tid``.
        """
        entry: EntryType = self.index.entry_index.get(tid)  # type: ignore
        if entry is None:
            raise KeyError(
                f"There is no entry with tid '{tid}'' in this datapack")
        return entry

    def get_ids_by_component(self, component: str) -> Set[str]:
        """
        Look up the component_index with key ``component``.
        """
        entry_set = self.index.component_index[component]
        if len(entry_set) == 0:
            logging.warning("There is no entry generated by '%s' "
                            "in this datapack", component)
        return entry_set

    def get_entries_by_component(self, component: str) -> Set[EntryType]:
        return {self.get_entry(tid)
                for tid in self.get_ids_by_component(component)}

    def get_ids_by_type(self, entry_type: Type[EntryType]) -> Set[str]:
        """
        Look up the type_index with key ``entry_type``.

        Returns:
             A set of entry tids. The entries are instances of entry_type (
             and also includes instances of the subclasses of entry_type).
        """
        subclass_index = set()
        for index_key, index_val in self.index.type_index.items():
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
            node: Union[str, EntryType],
            as_parent: bool
    ) -> Set[LinkType]:
        links: Set[LinkType] = set()
        if isinstance(node, Entry):
            tid = node.tid
            if tid is None:
                raise ValueError(f"The requested node has no tid. "
                                 f"Have you add this entry into the datapack?")
        else:
            tid = node

        if not self.index.link_index_on:
            self.index.build_link_index(self.links)

        for tid in self.index.link_index(tid, as_parent=as_parent):
            entry: EntryType = self.get_entry(tid)
            if self.validate_link(entry):
                links.add(entry)  # type: ignore
        return links

    def get_links_by_parent(
            self, parent: Union[str, EntryType]) -> Set[LinkType]:
        return self.get_links_from_node(parent, True)

    def get_links_by_child(self, child: Union[str, EntryType]) -> Set[LinkType]:
        return self.get_links_from_node(child, False)

    def get_groups_by_member(
            self, member: Union[str, EntryType]) -> Set[GroupType]:
        groups: Set[GroupType] = set()
        if isinstance(member, Entry):
            tid = member.tid
            if tid is None:
                raise ValueError(f"Argument member has no tid. "
                                 f"Have you add this entry into the datapack?")
        else:
            tid = member

        if not self.index.group_index_on:
            self.index.build_group_index(self.groups)

        for tid in self.index.group_index(tid):
            entry: EntryType = self.get_entry(tid)
            if self.validate_group(entry):
                groups.add(entry)  # type: ignore
        return groups


PackType = TypeVar('PackType', bound=BasePack)
