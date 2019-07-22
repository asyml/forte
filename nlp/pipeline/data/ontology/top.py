from abc import abstractmethod
from functools import total_ordering
from typing import Iterable, Optional, Set, Union, Type

from nlp.pipeline.utils import get_class_name


__all__ = [
    "Span",
    "Entry",
    "Annotation",
    "Link",
    "Group"
]


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
        component (str): the name of the engine that creates this entry,
            it should at least include the full engine name.
        tid (str, optional): a doc level universal id to identify
            this entry. If `None`, the DataPack will assign a unique id
            for this entry when we add the entry to the DataPack.
    """

    def __init__(self, component: str):
        self.component = component
        self.__tid: Optional[str] = None
        self.__data_pack = None

    @property
    def tid(self):
        return self.__tid

    def set_tid(self, tid: str):
        """Set the entry id"""
        self.__tid = f"{get_class_name(self)}.{tid}"

    @property
    def data_pack(self):
        return self.__data_pack

    def attach(self, data_pack):
        """Attach the entry itself to a data_pack"""
        self.__data_pack = data_pack

    def set_fields(self, **kwargs):
        """Set other entry fields"""
        for field_name, field_value in kwargs.items():
            if not hasattr(self, field_name):
                raise AttributeError(
                    f"class {get_class_name(self)}"
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

    def __init__(self, component: str, begin: int, end: int):
        super().__init__(component)
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
    parent_type: Type[Entry] = Annotation
    child_type: Type[Entry] = Annotation

    def __init__(self, component: str, parent: Optional[Entry] = None,
                 child: Optional[Entry] = None):
        super().__init__(component)
        self._parent: Optional[str] = None
        self._child: Optional[str] = None
        if parent is not None:
            self.set_parent(parent)
        if child is not None:
            self.set_child(child)

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

    def set_parent(self, parent: Entry):
        if not isinstance(parent, self.parent_type):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.parent_type}, but get {type(parent)}")

        self._parent = parent.tid

        if (self.data_pack is not None and
                self.data_pack.index.link_index_switch):
            self.data_pack.index.update_link_index()

    def set_child(self, child: Entry):
        if not isinstance(child, self.child_type):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.child_type}, but get {type(child)}")

        self._child = child.tid

        if (self.data_pack is not None and
                self.data_pack.index.link_index_switch):
            self.data_pack.index.update_link_index()

    def get_parent(self):
        """
        Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the parent of the link.
        """
        return self.data_pack.index.entry_index[self._parent]

    def get_child(self):
        """
        Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link.
        """
        return self.data_pack.index.entry_index[self._child]


class Group(Entry):
    """Group type entries, such as "coreference group". Each group has a set
    of members.
    """
    member_type: Type[Entry] = Annotation

    def __init__(self, component: str,
                 members: Optional[Set[Entry]] = None):

        super().__init__(component)
        self._members: Set[str] = set()
        if members is not None:
            self.add_members(members)

    def add_members(self, members: Union[Iterable[Entry], Entry]):
        """Add group members."""
        if not isinstance(members, Iterable):
            members = {members}

        for member in members:
            if not isinstance(member, self.member_type):
                raise TypeError(
                    f"The members of {type(self)} should be "
                    f"instances of {self.member_type}, but get {type(member)}")

            self._members.add(member.tid)

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

    def get_members(self):
        """
        Get the member entries in the group.

        Returns:
             An set of instances of :class:`Entry` that are the members of the
             group.
        """
        member_entries = set()
        for m in self.members:
            member_entries.add(self.data_pack.index.entry_index[m])
        return member_entries
