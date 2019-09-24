from abc import abstractmethod, ABC
from functools import total_ordering
from typing import Iterable, Optional, Set, Union, Tuple, Type, TypeVar, Any, \
    Hashable, Generic, Callable

from forte.common.exception import IncompleteEntryError
from forte.common.types import VectorType
from forte.utils import get_class_name, get_full_module_name
from forte.data.base_pack import PackType
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack

__all__ = [
    "Span",
    "Entry",
    "EntryType",
    "Annotation",
    "GroupType",
    "BaseGroup",
    "MultiPackGroup",
    "Group",
    "LinkType",
    "BaseLink",
    "Link",
    "MultiPackLink",
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


class Indexable(ABC):
    """
    A class that implement this would be indexable within the pack it lives in.
    """

    @property
    def index_key(self) -> Hashable:
        raise NotImplementedError


class Entry(Hashable, Indexable, Generic[PackType]):
    """
    The base class inherited by all NLP entries.
    There will be some associated attributes for each entry.
    - component: specify the creator of the entry
    - _data_pack: each entry can be attached to a pack with
        ``attach`` function.
    - _tid: a unique identifier of this entry in the data pack
    """

    def __init__(self, pack: PackType, embeddable: bool = False):
        self._tid: str

        self.__component: str
        self.__modified_fields: Set[str] = set()
        self._data_pack: PackType = pack

        self.__embeddable = embeddable

        self.__cached_embedding: Optional[VectorType] = None

        self.__embed_func: Optional[Callable[['Entry'], VectorType]] = None

    @property
    def tid(self):
        return self._tid

    @property
    def component(self):
        return self.__component

    def embed(self, recalculate: bool = False):
        """
        Get the embedding of this entry if it is embeddable.

        The internal embed function. User should implement :meth:``__embed()``
        instead.

        Args:
            recalculate: If True, recalculate the value and disregard the cached
            embedding.

        Returns:

        """
        if self.__embeddable:
            if self.__cached_embedding is None or recalculate:
                if self.__embed_func is None:
                    raise NotImplementedError(
                        'This embed function of the instance has not been '
                        'register, please register through '
                        ':meth:``register_embed_function``.')
                self.__embed_func(self)
            else:
                return self.__cached_embedding
        else:
            raise NotImplementedError(
                'You cannot call embed() for a non-embeddable entry.')

    def register_embed_function(self, func: Callable[['Entry'], VectorType]):
        """
        Implement this function to get the embedding.
        Returns:

        """
        self.__embed = func

    def __set_component(self, component: str):
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
    def data_pack(self) -> PackType:
        return self._data_pack

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

    def __eq__(self, other):
        if other is None:
            return False

        return (type(self), self._tid) == (type(other), other.tid)

    def __hash__(self) -> int:
        return hash((type(self), self._tid))

    @property
    def index_key(self) -> Hashable:
        return self._tid


EntryType = TypeVar('EntryType', bound=Entry)


@total_ordering
class Annotation(Entry[DataPack]):
    """Annotation type entries, such as "token", "entity mention" and
    "sentence". Each annotation has a text span corresponding to its offset
    in the text.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack)
        self._span = Span(begin, end)

    @property
    def span(self):
        return self._span

    def set_span(self, begin: int, end: int):
        self._span = Span(begin, end)

    def __hash__(self):
        return hash(
            (type(self), self.data_pack, self.span.begin, self.span.end)
        )

    def __eq__(self, other):
        if other is None:
            return False
        return (type(self), self.span.begin, self.span.end) == \
               (type(other), other.span.begin, other.span.end)

    def __lt__(self, other):
        """
        Have to support total ordering and be consistent with
        __eq__(self, other)
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

    @property
    def index_key(self) -> str:
        return self.tid


class BaseLink(Entry, ABC):
    def __init__(
            self,
            pack: PackType,
            parent: Optional[EntryType] = None,
            child: Optional[EntryType] = None
    ):
        super().__init__(pack)

        if parent is not None:
            self.set_parent(parent)
        if child is not None:
            self.set_child(child)

    @abstractmethod
    def set_parent(self, parent: EntryType):
        """
        This will set the `parent` of the current instance with given Entry
        The parent is saved internally by its pack specific index key.

        Args:
            parent: The parent entry.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def set_child(self, child: EntryType):
        """
        This will set the `child` of the current instance with given Entry
        The child is saved internally by its pack specific index key.

        Args:
            child: The child entry

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def get_parent(self) -> EntryType:
        """
        Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link
             from the given DataPack
        """
        raise NotImplementedError

    @abstractmethod
    def get_child(self) -> EntryType:
        """
        Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link
             from the given DataPack
        """
        raise NotImplementedError

    def __eq__(self, other):
        if other is None:
            return False
        return (type(self), self.get_parent(), self.get_child()) == \
               (type(other), other.get_parent(), other.get_child())

    def __hash__(self):
        return hash((type(self), self.get_parent(), self.get_child()))

    @property
    def index_key(self) -> str:
        return self.tid


LinkType = TypeVar('LinkType', bound=BaseLink)


class Link(BaseLink):
    """
    The Link type entry connects two entries, such as "dependency link", which
    connect two words and specifies its dependency label.  Each link has a
    parent node and a child node.
    """
    ParentType: Type[Entry] = Entry
    ChildType: Type[Entry] = Entry

    def __init__(
            self,
            pack: DataPack,
            parent: Optional[Entry] = None,
            child: Optional[Entry] = None
    ):
        self._parent: Optional[str] = None
        self._child: Optional[str] = None
        super().__init__(pack, parent, child)

    def set_parent(self, parent: ParentType):
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

        if (self.data_pack is not None and
                self.data_pack.index.link_index_switch):
            self.data_pack.index.add_link_parent(parent, self)

    def set_child(self, child: ChildType):
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

        if (self.data_pack is not None and
                self.data_pack.index.link_index_switch):
            self.data_pack.index.add_link_child(child, self)

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

    def get_parent(self) -> ParentType:
        """
        Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the parent of the link.
        """
        if self.data_pack is None:
            raise ValueError(f"Cannot get parent because link is not "
                             f"attached to any data pack.")
        return self.data_pack.get_entry_by_id(self._parent)

    def get_child(self) -> ChildType:
        """
        Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link.
        """
        if self.data_pack is None:
            raise ValueError(f"Cannot get child because link is not"
                             f" attached to any data pack.")
        return self.data_pack.get_entry_by_id(self._child)


class BaseGroup(Entry):
    """
    Group is an entry that represent a group of other entries. For example,
    a "coreference group" is a group of coreferential entities. Each group will
    store a set of members, no duplications allowed.

    This is the BaseGroup interface. Specific member constraints are defined
    in the inherited classes.
    """
    member_type: Type[Entry] = Entry  # type: ignore

    def __init__(
            self,
            pack: PackType,
            members: Optional[Set[EntryType]] = None,
    ):
        super().__init__(pack)

        # Store the group member's id.
        self._members: Set[str] = set()
        if members is not None:
            self.add_members(members)

    def add_member(self, member: Entry):
        """
        Add one entry to the group.
        Args:
            member:

        Returns:

        """
        self.add_members([member])

    def add_members(self, members: Iterable[Entry]):
        """
        Add members to the group.

        Args:
            members: An iterator of members to be added to the group.

        Returns:

        """
        for member in members:
            if not isinstance(member, self.member_type):
                raise TypeError(
                    f"The members of {type(self)} should be "
                    f"instances of {self.member_type}, but get {type(member)}")

            self._members.add(member.tid)

        if (self.data_pack is not None and
                self.data_pack.index.group_index_switch):
            # TODO: NO way, don't do all update.
            self.data_pack.index.update_group_index([self])

    @property
    def members(self):
        """
        A list of member tids. To get the member objects, call
        :meth:`get_members` instead.
        :return:
        """
        return self._members

    def __hash__(self):
        return hash((type(self), tuple(self.members)))

    def __eq__(self, other):
        if other is None:
            return False
        return (type(self), self.members) == (type(other), other.members)

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

    @property
    def index_key(self) -> str:
        return self.tid


GroupType = TypeVar("GroupType", bound=BaseGroup)


class Group(BaseGroup):
    """
    Group is an entry that represent a group of other entries. For example,
    a "coreference group" is a group of coreferential entities. Each group will
    store a set of members, no duplications allowed.
    """
    member_type: Type[Entry] = Entry  # type: ignore

    def __init__(
            self,
            pack: DataPack,
            members: Optional[Set[Entry]] = None,
    ):
        super().__init__(pack, members)


class SubEntry(Entry[MultiPack]):
    """
    This is used to identify an Entry in one of the packs in the Multipack.
    For example, the sentence in one of the packs. A pack_index and an entry
    is needed to identify this.

    Args:
        pack_index: Indicate which pack this entry belongs. If this is less
        than 0, then this is a cross pack entry.
        entry: The entry itself.
    """

    def __init__(self, pack: MultiPack, pack_index: int, entry: Entry):
        super().__init__(pack)
        self._pack_index = pack_index
        self._entry = entry

    @staticmethod
    def from_id(data_pack: MultiPack, pack_index: int, entry_id: str):
        ent = data_pack._packs[pack_index].index.entry_index[entry_id]
        return SubEntry(data_pack, pack_index, ent)

    @property
    def pack_index(self):
        return self._pack_index

    @property
    def entry(self):
        return self._entry

    def __hash__(self):
        return hash((type(self), self._pack_index, self._entry))

    def __eq__(self, other):
        if other is None:
            return False
        return (type(self), self.pack_index, self.entry
                ) == (type(other), other.pack_index, other.entry)

    @property
    def index_key(self) -> Tuple[int, str]:
        return self._pack_index, self._entry.tid


class MultiPackLink(BaseLink):
    """
    The MultiPackLink are used to link entries in a MultiPack, which is designed
    to support cross pack linking, this can support applications such as
    sentence alignment and cross-document coreference. Each link should have
    a parent node and a child node. Note that the nodes are SubEntry(s), thus
    have one additional index on which pack it comes from.
    """

    ParentType: Type[SubEntry] = SubEntry
    ChildType: Type[SubEntry] = SubEntry

    def __init__(
            self,
            pack: MultiPack,
            parent: Optional[ParentType],
            child: Optional[ChildType],
    ):
        """

        Args:
            parent: The parent of the link, it should be a tuple of the name and
            an entry.
            child:
        """
        super().__init__(pack, parent, child)

        self._parent: Optional[Tuple[int, str]] = None
        self._child: Optional[Tuple[int, str]] = None

        if parent is not None:
            self.set_parent(parent)
        if child is not None:
            self.set_child(child)

    @property
    def parent(self) -> Tuple[int, str]:
        if self._parent is None:
            raise IncompleteEntryError("Parent is not set for this link.")
        return self._parent

    @property
    def child(self) -> Tuple[int, str]:
        if self._child is None:
            raise IncompleteEntryError("Child is not set for this link.")
        return self._child

    def set_parent(self, parent: ParentType):
        """
        This will set the `parent` of the current instance with given Entry
        The parent is saved internally as a tuple: pack_name and entry.tid

        Args:
            parent: The parent of the link. Multiple

        Returns:

        """
        if not isinstance(parent, self.ParentType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ParentType}, but get {type(parent)}")
        self._parent = parent.index_key

    def set_child(self, child: ChildType):
        if not isinstance(child, self.ChildType):
            raise TypeError(
                f"The parent of {type(self)} should be an "
                f"instance of {self.ChildType}, but get {type(child)}")
        self._child = child.index_key

    def get_parent(self) -> EntryType:
        """
        Get the parent entry of the link.

        Returns:
             An instance of :class:`SubEntry` that is the parent of the link
             from the given DataPack.
        """
        if self._parent is None:
            raise IncompleteEntryError("The parent of this link is not set.")
        pack_idx, parent_tid = self._parent

        return SubEntry.from_id(self.data_pack, pack_idx, parent_tid)

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

        return SubEntry.from_id(self.data_pack, pack_idx, child_tid)


class MultiPackGroup(BaseGroup):
    """
    Group type entries, such as "coreference group". Each group has a set
    of members.
    """

    def __init__(
            self,
            pack: MultiPack,
            members: Optional[Set[SubEntry]],
    ):
        super().__init__(pack, members)
