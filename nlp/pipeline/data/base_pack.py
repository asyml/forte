import copy
import logging
from abc import abstractmethod
from collections import defaultdict

from typing import (
    Dict, Iterator, List, Optional, Type, Union, Any, Iterable,
    TypeVar, DefaultDict, Set, Generic)
from nlp.pipeline.data.ontology import (
    Entry, EntryType, Annotation, Span, Link, Group)

logger = logging.getLogger(__name__)

__all__ = [
    "BasePack",
    "BaseMeta",
    "InternalMeta",
    "BaseIndex",
    "PackType"
]


class BaseMeta:
    """
    Basic Meta information for both
    :class:`~nlp.pipeline.data.data_pack.DataPack` and
    :class:`~nlp.pipeline.data.multi_pack.MultiPack`.
    """
    def __init__(self, doc_id: Optional[str] = None):
        self.doc_id = doc_id
        # the pack has been processed by which processor in the pipeline
        self.process_state = ''
        # the pack has been chached by which processor in the pipeline
        self.cache_state = ''


class InternalMeta:
    """
    The internal meta information of **one kind of entry** in a datapack.
    Record the entry fields created in the :class:`BasePack` and the entry
    counters.

    Note that the :attr:`intertal_metas` in :class:`BasePack` is a dict in
    which the keys are entries types and the values are objects of
    :class:`InternalMeta`.
    """
    def __init__(self):
        self.id_counter = 0
        self.fields_created = defaultdict(set)
        self.default_component = None


class BasePack:
    """
    The base class of :class:`~nlp.pipeline.data.data_pack.DataPack` and
    :class:`~nlp.pipeline.data.multi_pack.MultiPack`.

    Args:
        doc_id (str, optional): a string identifier of the pack.

    """
    def __init__(self, doc_id: Optional[str] = None):
        self.links: List[Link] = []
        self.groups: List[Group] = []

        self.meta: BaseMeta = BaseMeta(doc_id)

        self.index: BaseIndex = BaseIndex(self)
        self.internal_metas: \
            Dict[type, InternalMeta] = defaultdict(InternalMeta)

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self.meta, k, v)

    @abstractmethod
    def add_entry(self, entry: EntryType) -> EntryType:
        """
        Force add an :class:`~nlp.pipeline.data.ontology.top.Entry` object to
        the :class:`BasePack` object.
        Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`~nlp.pipeline.data.ontology.top.Entry`
                object to be added to the pack.

        Returns:
            The input entry itself
        """
        raise NotImplementedError

    @abstractmethod
    def delete_entry(self, entry: EntryType):
        """
        Delete an :class:`~nlp.pipeline.data.ontology.top.Entry` object from
        the :class:`BasePack`.

        Args:
            entry (Entry): An :class:`~nlp.pipeline.data.ontology.top.Entry`
                object to be deleted from the pack.

        """
        raise NotImplementedError

    @abstractmethod
    def add_or_get_entry(self, entry: EntryType) -> EntryType:
        """
        Try to add an :class:`~nlp.pipeline.data.ontology.top.Entry` object to
        the :class:`BasePack` object.
        If a same entry already exists, will return the existing entry
        instead of adding the new one. Note that we regard two entries as the
        same if their :meth:`~nlp.pipeline.data.ontology.top.Entry.eq` have
        the same return value, and users could
        override :meth:`~nlp.pipeline.data.ontology.top.Entry.eq` in their
        custom entry classes.

        Args:
            entry (Entry): An :class:`~nlp.pipeline.data.ontology.top.Entry`
                object to be added to the pack.

        Returns:
            If a same entry already exists, returns the existing
            entry. Otherwise, return the (input) entry just added.
        """
        raise NotImplementedError

    @abstractmethod
    def record_fields(self, fields: List[str], entry_type: Type[Entry],
                      component: str):
        """Record in the internal meta that the ``entry_type`` entires generated
        by ``component`` have ``fields``.

        If ``component`` is "_ALL_", we will record the ``fields`` for all
        entries of the type ``entry_type`` regardless of their component in
        the internal meta.
        """
        raise NotImplementedError

    @abstractmethod
    def get_data(
            self,
            context_type: Type[Annotation],
            requests: Optional[Dict[Type[Entry], Union[Dict, List]]] = None,
            offset: int = 0
    ) -> Iterator[Dict[str, Any]]:
        """
        Get data in a dict format.
        # TODO: will add docstrings after designing multipack get_data()
        """
        raise NotImplementedError

    @abstractmethod
    def get_entries(self,
                    entry_type: Type[EntryType],
                    range_annotation: Optional[Annotation] = None,
                    components: Optional[Union[str, List[str]]] = None
                    ) -> Iterable[EntryType]:
        raise NotImplementedError

    def get(self,
            entry_type: Type[EntryType],
            range_annotation: Optional[Annotation] = None,
            component: Optional[str] = None) -> Iterable[EntryType]:
        """
        Get ``entry_type`` entries from the span of ``range_annotation`` in the
        pack.

        Args:
            entry_type (type): The type of entries requested.
            range_annotation (Annotation, optional): The range of entries
                requested. If `None`, will return valid entries in the range of
                whole pack.
            component (str or list[str], optional): The components generated
                the entries. If `None`, will return valid entries
                generated by any component.
        """
        return self.get_entries(entry_type, range_annotation, component)

    def view(self):
        return copy.deepcopy(self)


PackType = TypeVar('PackType', bound=BasePack)


class BaseIndex(Generic[PackType]):
    """
    A set of indexes used in :class:`BasePack`:

    #. :attr:`entry_index`, the index from each tid to the corresponding entry
    #. :attr:`type_index`, the index from each type to the entries of
       that type
    #. :attr:`component_index`, the index from each component to the
       entries generated by that component
    #. :attr:`link_index`, the index from child
       (:attr:`link_index["child_index"]`)and parent
       (:attr:`link_index["parent_index"]`) nodes to links
    #. :attr:`group_index`, the index from group members to groups.

    Args:
        pack (BasePack): the pack this :class:`BaseIndex` is indexing
    """

    def __init__(self, pack):
        self.data_pack: PackType = pack
        # basic indexes (switches always on)
        self.entry_index: Dict[str, Entry] = dict()
        self.type_index: DefaultDict[Type, Set[str]] = defaultdict(set)
        self.component_index: DefaultDict[str, Set[str]] = defaultdict(set)
        # other indexes (built when first looked up)
        self._group_index = defaultdict(set)
        self._link_index: Dict[str, DefaultDict[str, set]] = dict()
        # indexing switches
        self._group_index_switch = False
        self._link_index_switch = False

    @property
    def link_index_switch(self):
        return self._link_index_switch

    def turn_link_index_switch(self, on: bool):
        self._link_index_switch = on

    @property
    def group_index_switch(self):
        return self._group_index_switch

    def turn_group_index_switch(self, on: bool):
        self._group_index_switch = on

    def link_index(self, tid: str, as_parent: bool = True) -> Set[str]:
        """
        Look up the link_index with key ``tid``.

        Args:
            tid (str): the tid of the entry being looked up.
            as_parent (bool): If `as_patent` is True, will look up
                :attr:`link_index["parent_index"]` and return the tids of links
                whose parent is `tid`. Otherwise,  will look up
                :attr:`link_index["child_index"]` and return the tids of links
                whose child is `tid`.

        Returns:
            A set of entry tids.
        """
        if not self._link_index_switch:
            self.update_link_index(self.data_pack.links)
        if as_parent:
            return self._link_index["parent_index"][tid]
        else:
            return self._link_index["child_index"][tid]

    def group_index(self, tid: str) -> Set[str]:
        """
        Look up the group_index with key ``tid``.

        Args:
            tid (str): the tid of the entry being looked up.

        Returns:
            A set of entry tids.
        """
        if not self._group_index_switch:
            self.update_group_index(self.data_pack.groups)
        return self._group_index[tid]

    def in_span(self,
                inner_entry: Union[str, Entry],
                span: Span) -> bool:
        """
        Check whether the ``inner_entry`` is within the given ``span``.
        Link entries are considered in a span if both the
        parent and the child are within the span. Group entries are
        considered in a span if all the members are within the span.

        Args:
            inner_entry (str or Entry): An
                :class:`~nlp.pipeline.data.ontology.top.Entry` object to be
                checked.
            span (Span): A :class:`Span` object to be checked.
        """

        if isinstance(inner_entry, str):
            inner_entry = self.entry_index[inner_entry]

        if isinstance(inner_entry, Annotation):
            inner_begin = inner_entry.span.begin
            inner_end = inner_entry.span.end
        elif isinstance(inner_entry, Link):
            child = inner_entry.get_child()
            parent = inner_entry.get_parent()
            inner_begin = min(child.span.begin, parent.span.begin)
            inner_end = max(child.span.end, parent.span.end)
        elif isinstance(inner_entry, Group):
            inner_begin = -1
            inner_end = -1
            for mem in inner_entry.get_members():
                if inner_begin == -1:
                    inner_begin = mem.span.begin
                inner_begin = min(inner_begin, mem.span.begin)
                inner_end = max(inner_end, mem.span.end)
        else:
            raise ValueError(
                f"Invalid entry type {type(inner_entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )
        return inner_begin >= span.begin and inner_end <= span.end

    def have_overlap(self,
                     entry1: Union[Annotation, str],
                     entry2: Union[Annotation, str]) -> bool:
        """Check whether the two annotations have overlap in span.

        Args:
            entry1 (str or Annotation): An :class:`Annotation` object to be
                checked, or the tid of the annotation.
            entry2 (str or Annotation): Another :class:`Annotation` object to be
                checked, or the tid of the annotation.
        """
        if isinstance(entry1, str):
            e = self.entry_index[entry1]
            if not isinstance(e, Annotation):
                raise TypeError(f"'entry1' should be an instance of Annotation,"
                                f" but get {type(e)}")
            entry1 = e

        if not isinstance(entry1, Annotation):
            raise TypeError(f"'entry1' should be an instance of Annotation,"
                            f" but get {type(entry1)}")

        if isinstance(entry2, str):
            e = self.entry_index[entry2]
            if not isinstance(e, Annotation):
                raise TypeError(f"'entry2' should be an instance of Annotation,"
                                f" but get {type(e)}")
            entry2 = e

        if not isinstance(entry2, Annotation):
            raise TypeError(f"'entry2' should be an instance of Annotation,"
                            f" but get {type(entry2)}")

        return not (entry1.span.begin >= entry2.span.end or
                    entry1.span.end <= entry2.span.begin)

    def update_basic_index(self, entries: List[Entry]):
        """Update the basic indexes, including :attr:`entry_index`,
        :attr:`type_index`, and :attr:`component_index`.
        Args:
            entries (list): a list of entires to be added into the basic index.
        """
        for entry in entries:
            self.entry_index[entry.tid] = entry
            self.type_index[type(entry)].add(entry.tid)
            self.component_index[entry.component].add(entry.tid)

    def update_link_index(self, links: List[Link]):
        """
        Build or update :attr:`link_index`, the index from child and parent
        nodes to links. :attr:`link_index` consists of two sub-indexes:
        :attr:`link_index["child_index"]` is the index from child nodes
        to their corresponding links, and :attr:`link_index["parent_index"]`
        is the index from parent nodes to their corresponding links.

        Args:
            links (list): a list of links to be added into the index.
        """
        logger.debug("Updating link index")

        # if the link index hasn't been built, build it with all existing links
        if not self.link_index_switch:
            self.turn_link_index_switch(on=True)
            self._link_index["child_index"] = defaultdict(set)
            self._link_index["parent_index"] = defaultdict(set)
            links = self.data_pack.links

        # update the link index with new links
        for link in links:
            self._link_index["child_index"][link.child].add(link.tid)
            self._link_index["parent_index"][link.parent].add(link.tid)

    def update_group_index(self, groups: List[Group]):
        """
        Build or update :attr:`group_index`, the index from group members
        to groups.

        Args:
            groups (list): a list of groups to be added into the index.
        """
        logger.debug("Updating group index")

        # if the group index hasn't been built,
        # build it with all existing groups
        if not self.group_index_switch:
            self.turn_group_index_switch(on=True)
            self._group_index = defaultdict(set)
            groups = self.data_pack.groups

        # update the group index with new groups
        for group in groups:
            for member in group.members:
                self._group_index[member].add(group.tid)
