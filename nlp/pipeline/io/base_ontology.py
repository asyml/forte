""" This class defines the basic ontology supported by our system
"""
from abc import abstractmethod
from functools import total_ordering
from typing import Iterable

# TODO: recording the links in the parent and child seems to be more natural
@total_ordering
class Span:
    """
    A class recording the span of annotations. :class:`Span` objects could
    be totally ordered according to their :attr:`begin` and :attr:`end` fields.
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
            this entry.
    """

    def __init__(self, component: str, tid: str = None):
        self.tid = f"{self.__class__.__name__}.{tid}" if tid else None
        self.component = component

    def set_tid(self, tid: str):
        """Set the entry id"""
        self.tid = f"{self.__class__.__name__}.{tid}"

    def set_fields(self, **kwargs):
        """Set other entry fields"""
        for field_name, field_value in kwargs.items():
            if not hasattr(self, field_name):
                raise AttributeError(
                    f"class {self.__class__.__qualname__}"
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
        self.eq(other)"""
        if self.span != other.span:
            return self.span < other.span
        if self.component != other.component:
            return self.component < other.component
        return str(type(self)) < str(type(other))


class Link(Entry):
    """Link type entries, such as "predicate link". Each link has a parent node
    and a child node.
    """
    def __init__(self, component: str, tid: str = None):
        super().__init__(component, tid)
        self.parent = None
        self.child = None

    def hash(self):
        return hash((self.component, type(self), self.parent, self.child))

    def eq(self, other):
        return (type(self), self.component, self.parent, self.child) == \
               (type(other), other.component, other.parent, other.child)


class Group(Entry):
    """Group type entries, such as "coreference group". Each group has a set
    of members.
    """
    def __init__(self, component: str, tid: str = None):
        super().__init__(component, tid)
        self.members = set()

    def add_members(self, members: Iterable):
        """Add group members."""
        if not isinstance(members, Iterable):
            members = [members]
        self.members.update(members)

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

        def __init__(self, component: str, tid: str = None):
            super().__init__(component, tid)
            self.arg_type = None

    class PredicateMention(Annotation):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)
            # self.links = []

        # def add_link(self, link: Link):
        #     self.links.append(link)

    class CoreferenceGroup(Group):
        member_type = "CoreferenceMention"

        def __init__(self, component: str, tid: str = None):
            super().__init__(component, tid)
            self.coref_type = None

    class CoreferenceMention(Annotation):
        def __init__(self, component: str, begin: int, end: int,
                     tid: str = None):
            super().__init__(component, begin, end, tid)
