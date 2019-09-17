from abc import abstractmethod
from functools import total_ordering
from typing import Iterable, Optional, Set, Union, Type, TypeVar, Any

from forte.utils import get_class_name, get_full_module_name
from forte.data.base_pack import BasePack

__all__ = [
    "Span",
    "Entry",
    "EntryType",
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
    There will be some associated attributes for each entry.
    - component: specify the creator of the entry
    - _data_pack: each entry can be attached to a pack with
        ``attach`` function.
    - _tid: a unique identifier of this entry in the data pack
    """

    def __init__(self):
        self._data_pack: BasePack
        self._tid: str

        self.__component: str
        self.__modified_fields = set()

    @property
    def tid(self):
        return self._tid

    def __set_working_component(self, component: str):
        """
        Set the component of the creator of this entry.
        Args:
            component: The component name of the creator (processor or reader).

        Returns:

        """
        self.__component = component

    def set_tid(self, tid: str):
        """
        Set the entry tid.
        Args:
            tid: The entry tid.

        Returns:

        """
        self._tid = f"{get_full_module_name(self)}.{tid}"

    @property
    def data_pack(self):
        return self._data_pack

    def attach(self, data_pack: BasePack):
        """Attach the entry itself to a data_pack"""
        # TODO This may create cycle reference.
        self._data_pack = data_pack

    def set_fields(self, **kwargs):
        """Set other entry fields"""
        for field_name, field_value in kwargs.items():
            if not hasattr(self, field_name):
                raise AttributeError(
                    f"class {get_class_name(self)} "
                    f"has no attribute {field_name}"
                )
            setattr(self, field_name, field_value)
            self.__modified_fields.add(field_name)

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


EntryType = TypeVar('EntryType', bound=Entry)


@total_ordering
class Annotation(Entry):
    """Annotation type entries, such as "token", "entity mention" and
    "sentence". Each annotation has a text span corresponding to its offset
    in the text.
    """

    def __init__(self, begin: int, end: int):
        super().__init__()
        self._span = Span(begin, end)

    @property
    def span(self):
        return self._span

    def set_span(self, begin: int, end: int):
        self._span = Span(begin, end)

    def hash(self):
        return hash(
            (type(self), self.span.begin, self.span.end))

    def eq(self, other):
        return (type(self), self.span.begin, self.span.end) == \
               (type(other), other.span.begin, other.span.end)

    def __lt__(self, other):
        """Have to support total ordering and be consistent with
        eq(self, other)
        """
        if self.span != other.span:
            return self.span < other.span
        return str(type(self)) < str(type(other))

    @property
    def text(self):
        if self.data_pack is None:
            raise ValueError(f"Cannot get text because annotation is not "
                             f"attached to any data pack.")
        return self.data_pack.text[self.span.begin: self.span.end]


class Link(Entry):
    """Link type entries, such as "predicate link". Each link has a parent node
    and a child node.
    """
    parent_type: Type[Entry] = Entry  # type: ignore
    child_type: Type[Entry] = Entry  # type: ignore

    def __init__(self, parent: Optional[Entry] = None,
                 child: Optional[Entry] = None):
        super().__init__()
        self._parent: Any = None
        self._child: Any = None
        if parent is not None:
            self.set_parent(parent)
        if child is not None:
            self.set_child(child)

    def hash(self):
        return hash((type(self), self.parent, self.child))

    def eq(self, other):
        return (type(self), self.parent, self.child) == \
               (type(other), other.parent, other.child)

    @property
    def parent(self):
        """
        tid of the parent node. To get the object of the parent node, call
        :meth:`get_parent`.
        """
        return self._parent

    @property
    def child(self):
        """
        tid of the child node. To get the object of the child node, call
        :meth:`get_child`.
        """
        return self._child

    def set_parent(self, parent: Entry):
        if not isinstance(parent, self.parent_type):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.parent_type}, but get {type(parent)}")

        self._parent = parent.tid

        if (self.data_pack is not None and
                self.data_pack.index.link_index_switch):
            self.data_pack.index.update_link_index(links=[self])

    def set_child(self, child: Entry):
        if not isinstance(child, self.child_type):
            raise TypeError(
                f"The child of {type(self)} should be an "
                f"instance of {self.child_type}, but get {type(child)}")

        self._child = child.tid

        if (self.data_pack is not None and
                self.data_pack.index.link_index_switch):
            self.data_pack.index.update_link_index(links=[self])

    def get_parent(self):
        """
        Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the parent of the link.
        """
        if self.data_pack is None:
            raise ValueError(f"Cannot get parent because link is not "
                             f"attached to any data pack.")
        return self.data_pack.get_entry_by_id(self._parent)

    def get_child(self):
        """
        Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link.
        """
        if self.data_pack is None:
            raise ValueError(f"Cannot get child because link is not"
                             f" attached to any data pack.")
        return self.data_pack.get_entry_by_id(self._child)


class Group(Entry):
    """Group type entries, such as "coreference group". Each group has a set
    of members.
    """
    member_type: Type[Entry] = Entry  # type: ignore

    def __init__(self, members: Optional[Set[Entry]] = None):

        super().__init__()
        self._members: Set = set()
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
        """
        A list of member tids. To get the member objects, call
        :meth:`get_members` instead.
        :return:
        """
        return self._members

    def hash(self):
        return hash((type(self), tuple(self.members)))

    def eq(self, other):
        return (type(self), self.members) == \
               (type(other), other.members)

    def get_members(self):
        """
        Get the member entries in the group.

        Returns:
             An set of instances of :class:`Entry` that are the members of the
             group.
        """
        if self.data_pack is None:
            raise ValueError(f"Cannot get members because group is not "
                             f"attached to any data pack.")
        member_entries = set()
        for m in self.members:
            member_entries.add(self.data_pack.get_entry_by_id(m))
        return member_entries
