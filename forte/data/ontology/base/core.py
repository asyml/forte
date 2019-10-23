"""
Defines the basic data structures and interfaces for the Forte data
representation system.
"""

from abc import abstractmethod, ABC
from typing import (Iterable, Optional, Set, Type, Hashable, TypeVar, Generic)

from forte.data.container import ContainerType

__all__ = [
    "Entry",
    "BaseLink",
    "BaseGroup",
    "LinkType",
    "GroupType",
    "EntryType",
]


class Entry(Generic[ContainerType]):
    """
        The base class inherited by all NLP entries. This is the main data type
        for all in-text NLP analysis results. The main sub-types are
        ``Annotation``, ``Link`` and ``Group``.
        An :class:`forte.data.ontology.top.Annotation` object represents a
        span in text.
        A :class:`forte.data.ontology.top.Link` object represents a binary
        link relation between two entries.
        A :class:`forte.data.ontology.top.Group` object represents a
        collection of multiple entries.

        There will be some associated attributes for each entry, which can be
        set via :meth:`set_fields` and retrieved via :meth:`get_field`.

        Args:
            pack: Each entry should be associated with one pack upon creation.
    """

    def __init__(self, pack: ContainerType):
        super(Entry, self).__init__()

        self._tid: int = -1

        # The Entry should have a reference to the data pack, and the data pack
        # need to store the entries. In order to resolve the cyclic references,
        # we create a generic class EntryContainer to be the place holder of
        # the actual. Whether this entry can be added to the pack is delegated
        # to be checked by the pack.
        self.__pack: ContainerType = pack
        self.__field_modified: Set[str] = set()
        pack.validate(self)

    def __getstate__(self):
        """
        In serialization:
         - The pack is not serialize, and it will be set by the container.

        This also implies that it is not advised to serialize an entry on its
        own, without the Container as the context, there is little semantics
        remained in an entry.
        """
        state = self.__dict__.copy()
        state.pop('_Entry__pack')
        state.pop('_Entry__field_modified')
        return state

    def __setstate__(self, state):
        # Recover the internal __field_modified dict for the entry.
        # NOTE: the __pack will be set via set_pack from the Pack side.
        self.__dict__['_Entry__field_modified'] = set()
        self.__dict__.update(state)

    @property
    def tid(self) -> int:
        return self._tid

    def set_tid(self):
        """
        Set the entry id with the auto-increment manager
        """
        # self._tid = f"{get_full_module_name(self)}.{tid}"
        self._tid = self.pack.get_next_id()

    def attach(self, pack: ContainerType):
        self.__pack = pack

    @property
    def pack(self) -> ContainerType:
        return self.__pack

    def set_pack(self, pack: ContainerType):
        self.__pack = pack

    def set_fields(self, **kwargs):
        """
        Set the entry fields from the kwargs.

        Args:
            **kwargs: A set of key word arguments used to set the value. A key
            must be correspond to a field name of this entry, and a value must
            match the field's type.

        Returns:

        """
        for field_name, field_value in kwargs.items():
            # TODO: This is wrong, absence of attribute is treated the same as
            #  the attribute being None. We need to really identify
            #  whether the field exists to disallow users adding unknown fields.
            # if not hasattr(self, field_name):
            #     raise AttributeError(
            #         f"class {get_full_module_name(self)} "
            #         f"has no attribute {field_name}"
            #     )
            setattr(self, field_name, field_value)
            if self.tid == -1:
                # This means the entry is not part of any data pack yet, we
                # remember the field modification for now.
                self.__field_modified.add(field_name)
            else:
                # We add the record to the system.
                self.__pack.add_field_record(self.tid, field_name)

    def get_fields_modified(self) -> Set[str]:
        return self.__field_modified

    def reset_fields_modified(self):
        self.__field_modified.clear()

    def get_field(self, field_name):
        return getattr(self, field_name)

    def __eq__(self, other):
        """
        The eq function for :class:`Entry` objects.
        To be implemented in each subclass.
        """
        if other is None:
            return False

        return (type(self), self._tid) == (type(other), other.tid)

    def __hash__(self) -> int:
        """
        The hash function for :class:`Entry` objects.
        To be implemented in each subclass.
        """
        return hash((type(self), self._tid))

    @property
    def index_key(self) -> Hashable:
        return self._tid


EntryType = TypeVar("EntryType", bound=Entry)


class BaseLink(Entry, ABC):
    def __init__(
            self,
            pack: ContainerType,
            parent: Optional[Entry] = None,
            child: Optional[Entry] = None
    ):
        super().__init__(pack)

        if parent is not None:
            self.set_parent(parent)
        if child is not None:
            self.set_child(child)

    @abstractmethod
    def set_parent(self, parent: Entry):
        """
        This will set the `parent` of the current instance with given Entry
        The parent is saved internally by its pack specific index key.

        Args:
            parent: The parent entry.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def set_child(self, child: Entry):
        """
        This will set the `child` of the current instance with given Entry
        The child is saved internally by its pack specific index key.

        Args:
            child: The child entry

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def get_parent(self) -> Entry:
        """
        Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link
             from the given DataPack
        """
        raise NotImplementedError

    @abstractmethod
    def get_child(self) -> Entry:
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
    def index_key(self) -> int:
        return self.tid


class BaseGroup(Entry, Generic[EntryType]):
    """
    Group is an entry that represent a group of other entries. For example,
    a "coreference group" is a group of coreferential entities. Each group will
    store a set of members, no duplications allowed.

    This is the BaseGroup interface. Specific member constraints are defined
    in the inherited classes.
    """
    MemberType: Type[EntryType]

    def __init__(
            self,
            pack: ContainerType,
            members: Optional[Set[EntryType]] = None,
    ):
        super().__init__(pack)

        # Store the group member's id.
        self._members: Set[int] = set()
        if members is not None:
            self.add_members(members)

    def add_member(self, member: EntryType):
        """
        Add one entry to the group.

        Args:
            member:

        Returns:

        """
        self.add_members([member])

    def add_members(self, members: Iterable[EntryType]):
        """
        Add members to the group.

        Args:
            members: An iterator of members to be added to the group.

        Returns:

        """
        for member in members:
            if not isinstance(member, self.MemberType):
                raise TypeError(
                    f"The members of {type(self)} should be "
                    f"instances of {self.MemberType}, but get {type(member)}")

            self._members.add(member.tid)

    @property
    def members(self):
        """
        A list of member tids. To get the member objects, call
        :meth:`get_members` instead.
        :return:
        """
        return self._members

    def __hash__(self):
        """
        The hash function of :class:`Group`.

        Users can define their own hash function by themselves but this must
        be consistent to :meth:`eq`.
        """
        return hash((type(self), tuple(self.members)))

    def __eq__(self, other):
        """
        The eq function of :class:`Group`.
        By default, :class:`Group` objects are regarded as the same if
        they have the same type, members, and are generated by the same
        component.

        Users can define their own eq function by themselves but this must
        be consistent to :meth:`hash`.
        """
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
        if self.pack is None:
            raise ValueError(f"Cannot get members because group is not "
                             f"attached to any data pack.")
        member_entries = set()
        for m in self.members:
            member_entries.add(self.pack.get_entry(m))
        return member_entries

    @property
    def index_key(self) -> int:
        return self.tid


GroupType = TypeVar("GroupType", bound=BaseGroup)
LinkType = TypeVar('LinkType', bound=BaseLink)
