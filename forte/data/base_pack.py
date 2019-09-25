import copy
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import (DefaultDict, Dict, Generic, List, Optional, Set, Type,
                    Tuple, Hashable, TypeVar)

import jsonpickle

from forte.common.exception import PackIndexError
from forte.common.types import EntryType, LinkType, GroupType
from forte.data.ontology.top import (Entry, BaseGroup, BaseLink)
from forte.data.container import EntryContainer

logger = logging.getLogger(__name__)

__all__ = [
    "BasePack",
    "BaseMeta",
    "InternalMeta",
    "PackType"
]


class BaseMeta:
    """
    Basic Meta information for both DataPack and MultiPack.
    """

    def __init__(self, doc_id: Optional[str] = None):
        self.doc_id: Optional[str] = doc_id

        # TODO: These two are definitely internal.
        self.process_state: str = ''
        self.cache_state: str = ''


class InternalMeta:
    """
    The internal meta information of **one kind of entry** in a datapack.
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


class BasePack(EntryContainer):
    """
    The base class of data packages. Currently we support two types of data
    packages, DataPack and MultiPack.
    """

    def __init__(self, doc_id: Optional[str] = None):
        super().__init__()

        self.links: List[BaseLink] = []
        self.groups: List[BaseGroup] = []

        self.meta: BaseMeta = BaseMeta(doc_id)
        self.index: BaseIndex = BaseIndex()
        self.internal_metas: \
            Dict[type, InternalMeta] = defaultdict(InternalMeta)

        # This is used internally when a processor takes the ownership of this
        # DataPack.
        self.__owner_component: str = '__default__'
        self.__poison: bool = False

    def enter_processing(self, component_name: str):
        self.__owner_component = component_name

    def current_component(self):
        return self.__owner_component

    def exit_processing(self):
        self.__owner_component = '__default__'

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self.meta, k, v)

    @staticmethod
    def get_poison():
        """
            A poison is an object that used denote the end of a data stream.
            Internally, we use a special poison pack object to indicate there
            is no more data to consume by downstream.
        """
        pack = BasePack('__poison__')
        pack.set_as_poison()
        return pack

    def set_as_poison(self):
        self.__poison = True

    def is_poison(self) -> bool:
        """
            See :meth:``get_poison``.
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
        Add an :class:`Entry` object to the :class:`BasePack` object.
        Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the pack.

        Returns:
            The input entry itself
        """
        raise NotImplementedError

    @abstractmethod
    def add_or_get_entry(self, entry: EntryType) -> EntryType:
        """
        Try to add an :class:`Entry` object to the :class:`DataPack` object.
        If a same entry already exists, will return the existing entry
        instead of adding the new one. Note that we regard two entries to be
        same if their :meth:`eq` have the same return value, and users could
        override :meth:`eq` in their custom entry classes.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the pack.

        Returns:
            If a same entry already exists, returns the existing
            entry. Otherwise, return the (input) entry just added.
        """
        raise NotImplementedError

    @abstractmethod
    def record_fields(self, fields: List[str], entry_type: Type[Entry],
                      component: str):
        """Record in the internal meta that the ``entry_type`` generated by
        ``component`` have ``fields``.

        If ``component`` is "_ALL_", we will record ``fields`` for all existing
        components in the internal meta of ``entry_type``.
        """
        raise NotImplementedError

    def serialize(self) -> str:
        """
        Serializes a pack to a string.
        """
        return jsonpickle.encode(self, unpicklable=True)

    def view(self):
        return copy.deepcopy(self)


PackType = TypeVar('PackType', bound=BasePack)



