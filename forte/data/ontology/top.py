from functools import total_ordering
from typing import (Optional, Set, Tuple, Type)

from forte.common.exception import IncompleteEntryError
from forte.data.base_pack import PackType
from forte.data.ontology.core import Entry, BaseLink, BaseGroup
from forte.data.base import Span

__all__ = [
    "Annotation",
    "Group",
    "Link",
    "MultiPackGroup",
    "MultiPackLink",
    "SubEntry",
    "SinglePackEntries",
    "MultiPackEntries",
]


@total_ordering
class Annotation(Entry):
    """
    Annotation type entries, such as "token", "entity mention" and
    "sentence". Each annotation has a :class:`Span` corresponding to its offset
    in the text.

    Args:
        pack (PackType): The container that this annotation
         will be added to.
        begin (int): The offset of the first character in the annotation.
        end (int): The offset of the last character in the annotation + 1.
    """

    def __init__(self, pack: PackType, begin: int, end: int):
        super().__init__(pack)
        if begin > end:
            raise ValueError(
                f"The begin {begin} of span is greater than the end {end}")
        self._span = Span(begin, end)

    @property
    def span(self):
        return self._span

    def set_span(self, begin: int, end: int):
        """
        Set the span of the annotation.
        """
        if begin > end:
            raise ValueError(
                f"The begin {begin} of span is greater than the end {end}")
        self._span = Span(begin, end)

    def __hash__(self):
        """
        The hash function of :class:`Annotation`.

        Users can define their own hash function by themselves but this must
        be consistent to :meth:`eq`.
        """
        return hash(
            (type(self), self.pack, self.span.begin, self.span.end)
        )

    def __eq__(self, other):
        """
        The eq function of :class:`Annotation`.
        By default, :class:`Annotation` objects are regarded as the same if
        they have the same type, span, and are generated by the same component.

        Users can define their own eq function by themselves but this must
        be consistent to :meth:`hash`.
        """
        if other is None:
            return False
        return (type(self), self.span.begin, self.span.end) == \
               (type(other), other.span.begin, other.span.end)

    def __lt__(self, other):
        """
        To support total_ordering, :class:`Annotations` must provide
        :meth:`__lt__`.

        Users can define their own lt function by themselves but this must
        be consistent to :meth:`__eq__`.
        """
        if self.span != other.span:
            return self.span < other.span
        return str(type(self)) < str(type(other))

    @property
    def text(self):
        if self.pack is None:
            raise ValueError(f"Cannot get text because annotation is not "
                             f"attached to any data pack.")
        return self.pack.get_span_text(self.span)

    @property
    def index_key(self) -> int:
        return self.tid


class Link(BaseLink):
    """
    Link type entries, such as "predicate link". Each link has a parent node
    and a child node.

    Args:
         pack (EntryContainer): The container that this annotation
         will be added to.

        parent (Entry, optional): the parent entry of the link.
        child (Entry, optional): the child entry of the link.
    """
    ParentType: Type[Entry]
    ChildType: Type[Entry]

    def __init__(
            self,
            pack: PackType,
            parent: Optional[Entry] = None,
            child: Optional[Entry] = None
    ):
        self._parent: Optional[int] = None
        self._child: Optional[int] = None
        super().__init__(pack, parent, child)

    # TODO: Can we get better type hint here?
    def set_parent(self, parent: Entry):
        """
        This will set the `parent` of the current instance with given Entry
        The parent is saved internally by its pack specific index key.

        Args:
            parent: The parent entry.

        Returns:

        """
        if not isinstance(parent, self.ParentType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(parent)}")
        self._parent = parent.tid

    def set_child(self, child: Entry):
        """
       This will set the `child` of the current instance with given Entry
       The child is saved internally by its pack specific index key.

       Args:
           child: The child entry

        Args:
            child:

        Returns:

        """
        if not isinstance(child, self.ChildType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(child)}")
        self._child = child.tid

    @property
    def parent(self):
        """
        Get ``tid`` of the parent node. To get the object of the parent node,
        call :meth:`get_parent`.
        """
        return self._parent

    @property
    def child(self):
        """
        Get ``tid`` of the child node. To get the object of the child node,
        call :meth:`get_child`.
        """
        return self._child

    def get_parent(self) -> Entry:
        """
        Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the parent of the link.
        """
        if self.pack is None:
            raise ValueError(f"Cannot get parent because link is not "
                             f"attached to any data pack.")
        if self._parent is None:
            raise ValueError(f"The child of this entry is not set.")
        return self.pack.get_entry(self._parent)

    def get_child(self) -> Entry:
        """
        Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link.
        """
        if self.pack is None:
            raise ValueError(f"Cannot get child because link is not"
                             f" attached to any data pack.")
        if self._child is None:
            raise ValueError(f"The child of this entry is not set.")
        return self.pack.get_entry(self._child)


class Group(BaseGroup[Entry]):
    """
    Group is an entry that represent a group of other entries. For example,
    a "coreference group" is a group of coreferential entities. Each group will
    store a set of members, no duplications allowed.
    """
    MemberType: Type[Entry] = Entry


class SubEntry(Entry[PackType]):
    """
    This is used to identify an Entry in one of the packs in the
    :class:`Multipack`.
    For example, the sentence in one of the packs. A pack_index and an entry
    is needed to identify this.

    Args:
        pack_index: Indicate which pack this entry belongs. If this is less
        than 0, then this is a cross pack entry.
        entry_id: The tid of the entry in the sub pack.
    """

    def __init__(self, pack: PackType, pack_index: int, entry_id: int):
        super().__init__(pack)
        self._pack_index: int = pack_index
        self._entry_id: int = entry_id

    @property
    def pack_index(self):
        return self._pack_index

    @property
    def entry_id(self):
        return self._entry_id

    def __hash__(self):
        return hash((type(self), self._pack_index, self._entry_id))

    def __eq__(self, other):
        if other is None:
            return False
        return (type(self), self.pack_index, self.entry_id
                ) == (type(other), other.pack_index, other.entry)

    @property
    def index_key(self) -> Tuple[int, int]:
        return self._pack_index, self._entry_id


class MultiPackLink(BaseLink):
    """
    This is used to link entries in a :class:`MultiPack`, which is
    designed to support cross pack linking, this can support applications such
    as sentence alignment and cross-document coreference. Each link should have
    a parent node and a child node. Note that the nodes are SubEntry(s), thus
    have one additional index on which pack it comes from.
    """

    ParentType: Type[SubEntry]
    """The parent type of this link."""
    ChildType: Type[SubEntry]
    """The Child type of this link."""

    def __init__(
            self,
            pack: PackType,
            parent: Optional[SubEntry],
            child: Optional[SubEntry],
    ):
        """

        Args:
            parent: The parent of the link, it should be a tuple of the name and
            an entry.
            child:
        """
        super().__init__(pack, parent, child)

        self._parent: Optional[Tuple[int, int]] = None
        self._child: Optional[Tuple[int, int]] = None

        if parent is not None:
            self.set_parent(parent)
        if child is not None:
            self.set_child(child)

    @property
    def parent(self) -> Tuple[int, int]:
        if self._parent is None:
            raise IncompleteEntryError("Parent is not set for this link.")
        return self._parent

    @property
    def child(self) -> Tuple[int, int]:
        if self._child is None:
            raise IncompleteEntryError("Child is not set for this link.")
        return self._child

    def set_parent(self, parent: SubEntry):  # type: ignore
        """
        This will set the `parent` of the current instance with given Entry
        The parent is saved internally as a tuple: pack_name and entry.tid

        Args:
            parent: The parent of the link, identified as a sub entry, which
            has a value for the pack index and the tid in the pack.

        Returns:

        """
        if not isinstance(parent, self.ParentType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(parent)}")
        self._parent = parent.index_key

    def set_child(self, child: SubEntry):  # type: ignore
        if not isinstance(child, self.ChildType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ChildType}, but get {type(child)}")
        self._child = child.index_key

    def get_parent(self) -> SubEntry:
        """
        Get the parent entry of the link.

        Returns:
             An instance of :class:`SubEntry` that is the parent of the link
             from the given DataPack.
        """
        if self._parent is None:
            raise IncompleteEntryError("The parent of this link is not set.")
        pack_idx, parent_tid = self._parent

        return SubEntry(self.pack, pack_idx, parent_tid)

    def get_child(self) -> SubEntry:
        """
        Get the child entry of the link.

        Returns:
             An instance of :class:`SubEntry` that is the child of the link
             from the given DataPack.
        """
        if self._child is None:
            raise IncompleteEntryError("The parent of this link is not set.")

        pack_idx, child_tid = self._child

        return SubEntry(self.pack, pack_idx, child_tid)


class MultiPackGroup(BaseGroup[SubEntry]):
    """
    Group type entries, such as "coreference group". Each group has a set
    of members.
    """

    def __init__(
            self,
            pack: PackType,
            members: Optional[Set[SubEntry]],
    ):  # pylint: disable=useless-super-delegation
        super().__init__(pack, members)


SinglePackEntries = (Link, Group, Annotation)
MultiPackEntries = (MultiPackLink, MultiPackGroup)
