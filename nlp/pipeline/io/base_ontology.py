"""
This class defines the basic ontology supported by our system
"""
from abc import abstractmethod
from functools import total_ordering
from typing import Iterable, Set, Union
from nlp.pipeline.utils import *


@total_ordering
class Span:
    """
    A class recording the span of annotations. :class:`Span` objects could
    be totally ordered according to their :attr:`begin` as the first sort key
    and :attr:`end` as the second sort key.
    """

    def __init__(self, begin: int, end: int):
        self.begin = begin
        self.end = end

    def __lt__(self, other):
        if self.begin == other.begin:
            return self.end < other.end
        return self.begin < other.begin

    def __eq__(self, other):
        return (self.begin, self.end) == (other.begin, other.end)


class Entry:
    """The base class inherited by all NLP entries.

    Args:
        component (str): the engine name that creates this entry,
            it should at least include the full engine name.
        tid (str, optional): a doc level universal id to identify
            this entry. If `None`, the DataPack will assign a unique id
            for this entry when we add the entry to the DataPack.
    """

    def __init__(self, component: str, tid: str = None):
        self.tid = f"{get_class_name(self, lower=True)}.{tid}" if tid else None
        self.component = component
        self.data_pack = None

    def set_tid(self, tid: str):
        """Set the entry id"""
        self.tid = f"{get_class_name(self, lower=True)}.{tid}"

    def set_fields(self, **kwargs):
        """Set other entry fields"""
        for field_name, field_value in kwargs.items():
            if not hasattr(self, field_name):
                raise AttributeError(
                    f"class {get_qual_name(self, lower=True)}"
                    f" has no attribute {field_name}"
                )
            setattr(self, field_name, field_value)

    @abstractmethod
    def hash(self):
        """The hash function for :class:`Entry` objects.
        To be implemented in each subclass."""
        raise NotImplementedError

    @abstractmethod
    def eq(self, other):
        """The eq function for :class:`Entry` objects.
        To be implemented in each subclass."""
        raise NotImplementedError

    def __hash__(self):
        return self.hash()

    def __eq__(self, other):
        return self.eq(other)


@total_ordering
class Annotation(Entry):
    """Annotation type entries, such as "token", "entity mention" and
    "sentence". Each annotation has a text span corresponding to its offset
    in the text.
    """

    def __init__(self, component: str, begin: int, end: int, tid: str = None):
        super().__init__(component, tid)
        self.span = Span(begin, end)

    def hash(self):
        return hash(
            (self.component, type(self), self.span.begin, self.span.end))

    def eq(self, other):
        return (type(self), self.component, self.span.begin, self.span.end) == \
               (type(other), other.component, other.span.begin, other.span.end)

    def __lt__(self, other):
        """Have to support total ordering and be consistent with
        eq(self, other)
        """
        if self.span != other.span:
            return self.span < other.span
        if self.component != other.component:
            return self.component < other.component
        return str(type(self)) < str(type(other))

    @property
    def text(self):
        return self.data_pack.text[self.span.begin: self.span.end]


class Link(Entry):
    """Link type entries, such as "predicate link". Each link has a parent node
    and a child node.
    """
    parent_type = None
    child_type = None

    def __init__(self, component: str, parent_id: str = None,
                 child_id: str = None, tid: str = None):
        super().__init__(component, tid)
        self._parent = parent_id
        self._child = child_id

    def hash(self):
        return hash((self.component, type(self), self.parent, self.child))

    def eq(self, other):
        return (type(self), self.component, self.parent, self.child) == \
               (type(other), other.component, other.parent, other.child)

    @property
    def parent(self):
        return self._parent

    @property
    def child(self):
        return self._child

    @parent.setter
    def parent(self, parent_id: str):
        if self.parent_type is None or self.data_pack is None:
            self._parent = parent_id
        else:
            parent_entry = self.data_pack.index.entry_index[parent_id]
            class_name = get_class_name(parent_entry, lower=True)
            if class_name == self.parent_type:
                self._parent = parent_id
            else:
                raise TypeError(
                    f"The parent of {get_qual_name(self)} should be an "
                    f"instance of {self.parent_type}, but get {class_name}")

        if (self.data_pack is not None and
                self.data_pack.index.link_index_switch):
            self.data_pack.index.update_link_index()

    @child.setter
    def child(self, child_id: str):
        if self.child_type is None or self.data_pack is None:
            self._child = child_id
        else:
            child_entry = self.data_pack.index.entry_index[child_id]
            class_name = get_class_name(child_entry, lower=True)
            if class_name == self.child_type:
                self._child = child_id
            else:
                raise TypeError(
                    f"The child of {get_qual_name(self)} should be an "
                    f"instance of {self.child_type}, but get {class_name}")

        if (self.data_pack is not None and
                self.data_pack.index.link_index_switch):
            self.data_pack.index.update_link_index()


class Group(Entry):
    """Group type entries, such as "coreference group". Each group has a set
    of members.
    """
    member_type = None

    def __init__(self, component: str, members: Set[str] = None,
                 tid: str = None):
        super().__init__(component, tid)
        self._members = set(members) if members else set()

    def add_members(self, members: Union[Iterable, str]):
        """Add group members."""
        if isinstance(members, str):
            members = [members]
        self.members.update(members)

        if (self.data_pack is not None and
                self.data_pack.index.group_index_switch):
            self.data_pack.index.update_group_index([self])

    @property
    def members(self):
        return self._members

    def hash(self):
        return hash((type(self), self.component, tuple(self.members)))

    def eq(self, other):
        return (type(self), self.component, self.members) == \
               (type(other), other.component, other.members)


class Meta:
    """Meta information of a document.
    """

    def __init__(self, doc_id: str = None):
        self.doc_id = doc_id


class BaseOntology:
    """The basic ontology that could be inherited by other more specific
     ontology"""

    class Token(Annotation):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)

    class Sentence(Annotation):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)

    class EntityMention(Annotation):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)
            self.ner_type = None

    class PredicateArgument(Annotation):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)

    class PredicateLink(Link):
        parent_type = "PredicateMention"
        child_type = "PredicateArgument"

        def __init__(self, component: str, parent_id: str = None,
                     child_id: str = None, tid: str = None):
            super().__init__(component, parent_id, child_id, tid)
            self.arg_type = None

    class PredicateMention(Annotation):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)

    class CoreferenceGroup(Group):
        member_type = "CoreferenceMention"

        def __init__(self, component: str, tid: str = None):
            super().__init__(component, tid)
            self.coref_type = None

    class CoreferenceMention(Annotation):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)
