# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Defines the basic data structures and interfaces for the Forte data
representation system.
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import (
    Iterable, Optional, Set, Type, Hashable, TypeVar, Generic, MutableSequence,
    Union, Dict, MutableMapping, Iterator, get_type_hints, Any)

import numpy as np

from forte.common import PackDataException
from forte.data.container import ContainerType, BasePointer

__all__ = [
    "Entry",
    "BaseLink",
    "BaseGroup",
    "LinkType",
    "GroupType",
    "EntryType",
    "Pointer",
    "MpPointer",
    "FDict",
    "FList",
    "MultiEntry"
]

from forte.utils.utils import check_type

default_entry_fields = [
    '_Entry__pack', '_tid', '_embedding', '_span', '_parent', '_child',
    '_members', '_Entry__field_modified', 'field_records', 'creation_records',
    '_id_manager']


@dataclass
class Entry(Generic[ContainerType]):
    r"""The base class inherited by all NLP entries. This is the main data type
    for all in-text NLP analysis results. The main sub-types are
    ``Annotation``, ``Link`` and ``Group``.

    An :class:`forte.data.ontology.top.Annotation` object represents a
    span in text.

    A :class:`forte.data.ontology.top.Link` object represents a binary
    link relation between two entries.

    A :class:`forte.data.ontology.top.Group` object represents a
    collection of multiple entries.

    Attributes:
        self.embedding: The embedding vectors (numpy array of floats) of this
            entry.

    Args:
        pack: Each entry should be associated with one pack upon creation.
    """

    def __init__(self, pack: ContainerType):
        # The Entry should have a reference to the data pack, and the data pack
        # need to store the entries. In order to resolve the cyclic references,
        # we create a generic class EntryContainer to be the place holder of
        # the actual. Whether this entry can be added to the pack is delegated
        # to be checked by the pack.
        super().__init__()
        self.__pack: ContainerType = pack
        self._tid: int = self.pack.get_next_id()
        self._embedding: np.ndarray = np.empty(0)
        self.pack.validate(self)
        self.pack.record_new_entry(self)
        # self.__field_modified: Set[str] = set()

    def regret_creation(self):
        self.__pack.regret_record(self)

    def __getstate__(self):
        r"""In serialization, the pack is not serialize, and it will be set
        by the container.

        This also implies that it is not advised to serialize an entry on its
        own, without the ``Container`` as the context, there is little semantics
        remained in an entry.
        """
        state = self.__dict__.copy()
        # During serialization, convert the numpy array as a list.
        state["_embedding"] = self._embedding.tolist()
        state.pop('_Entry__pack')
        # state.pop('_Entry__field_modified')
        return state

    def __setstate__(self, state):
        # Recover the internal __field_modified dict for the entry.
        # NOTE: the __pack will be set via set_pack from the Pack side.
        # self.__dict__['_Entry__field_modified'] = set()

        # During de-serialization, convert the list back to numpy array.
        state["_embedding"] = np.array(state["_embedding"])
        self.__dict__.update(state)

    # using property decorator
    # a getter function for self._embedding
    @property
    def embedding(self):
        r"""Get the embedding vectors (numpy array of floats) of the entry.
        """
        return self._embedding

    # a setter function for self._embedding
    @embedding.setter
    def embedding(self, embed):
        r"""Set the embedding vectors of the entry.

        Args:
            embed: The embedding vectors which can be numpy array of floats or
                list of floats.
        """
        self._embedding = np.array(embed)

    @property
    def tid(self) -> int:
        """
        Get the id of this entry.

        Returns:

        """
        return self._tid

    @property
    def pack(self) -> ContainerType:
        return self.__pack

    @property
    def pack_id(self) -> int:
        """
        Get the id of the pack that contains this entry.

        Returns:

        """
        return self.__pack.meta.pack_id  # type: ignore

    def set_pack(self, pack: ContainerType):
        self.__pack = pack

    def create_pointer(self, from_entry: "Entry"):
        """
        Create a pointer of this entry relative to the ``from_entry``.

        Args:
            from_entry: The entry to point from.

        Returns:
             A pointer to the this entry from the ``from_entry``.
        """
        if isinstance(from_entry, MultiEntry):
            return MpPointer(
                from_entry.pack.get_pack_index(self.pack_id),
                self.tid
            )
        elif isinstance(from_entry, Entry):
            return Pointer(self.tid)

    def from_pointer(self, ptr: BasePointer):
        """
        Get the entry via the pointer from this entry.

        Args:
            ptr:

        Returns:

        """
        return self.pack.get_entry(ptr)

    def _check_attr_type(self, key, value):
        """
        Use the type hint to validate whether the provided value is as expected.

        Args:
            key:  The field name.
            value: The field value.

        Returns:

        """
        if key not in default_entry_fields:
            hints = get_type_hints(self.__class__)
            is_valid = check_type(value, hints[key])
            if not is_valid:
                raise TypeError(
                    f"The [{key}] attribute of [{type(self)}] "
                    f"should be [{hints[key]}], but got [{type(value)}].")

    def __setattr__(self, key, value):
        self._check_attr_type(key, value)

        if isinstance(value, Entry):
            if value.pack == self.pack:
                # Save a pointer to the value from this entry.
                self.__dict__[key] = Pointer(value.tid)
            else:
                raise PackDataException(
                    "An entry cannot refer to entries in another data pack.")
        else:
            super().__setattr__(key, value)

        # We add the record to the system.
        if key not in default_entry_fields:
            self.__pack.add_field_record(self.tid, key)

    def __getattribute__(self, item):
        v = super().__getattribute__(item)
        if isinstance(v, BasePointer):
            # Using the pointer to get the entry.
            return self.pack.get_entry(v)
        else:
            return v

    # TODO: Will replaced by = assignment
    def set_fields(self, **kwargs):
        r"""Set the entry fields from the kwargs.

        Args:
            **kwargs: A set of key word arguments used to set the value. A key
            must be correspond to a field name of this entry, and a value must
            match the field's type.
        """
        for field_name, field_value in kwargs.items():
            if field_name in vars(self):
                # NOTE: hasattr does not work here because it check both
                #  functions and attributes. We are only interested to see if
                #  the attributes are there.
                #  For example, if we use hasattr, is will return True for
                #  the setter and getter of the attribute name.

                # TODO: convert field value to integer automatically here.
                if isinstance(field_value, Entry):
                    setattr(self, field_name, field_value.tid)
                else:
                    setattr(self, field_name, field_value)
            else:
                raise AttributeError(
                    f"The entry type [{self.__class__}] does not have an "
                    f"attribute: '{field_name}'.")

            # We add the record to the system.
            self.__pack.add_field_record(self.tid, field_name)

    def __eq__(self, other):
        r"""The eq function for :class:`Entry` objects.
        To be implemented in each subclass.
        """
        if other is None:
            return False

        return (type(self), self._tid) == (type(other), other.tid)

    def __lt__(self, other):
        r"""Comparison based on type and id.
        """
        return (str(type(self)), self._tid) < (str(type(other)), other.tid)

    def __hash__(self) -> int:
        r"""The hash function for :class:`Entry` objects.
        To be implemented in each subclass.
        """
        return hash((type(self), self._tid))

    @property
    def index_key(self) -> Hashable:
        # Think about how to use the index key carefully.
        return self._tid


class MultiEntry(Entry, ABC):
    def __setattr__(self, key, value):
        """
        Handle the special sub-entry case in the multi pack case.

        Args:
            key:
            value:

        Returns:

        """
        self._check_attr_type(key, value)

        if isinstance(value, Entry):
            # Save a pointer to the value from this entry.
            self.__dict__[key] = value.create_pointer(self)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, item):
        """
        Handle the special sub-entry case in the multi pack case.

        Returns:

        """
        v = super().__getattribute__(item)

        if isinstance(v, MpPointer):
            # Using the multi pack pointer to get the entry.
            return self.pack.get_entry(v)
        else:
            return v

    def create_pointer(self, from_entry: "Entry"):
        """
        Get a pointer of the entry relative to this entry

        Args:
            from_entry: The entry relative from.

        Returns:
             A pointer relative to the this entry.
        """
        if isinstance(from_entry, MultiEntry):
            return Pointer(self.tid)
        elif isinstance(from_entry, Entry):
            raise ValueError(
                "Do not support reference a multi pack entry from an entry.")


EntryType = TypeVar("EntryType", bound=Entry)

ParentEntryType = TypeVar("ParentEntryType", bound=Entry)


class FList(Generic[ParentEntryType], MutableSequence):
    """
    FList allows the elements to be Forte entries. FList will internally
    stores the entry as their tid to avoid nesting.
    """

    def __init__(self, parent_entry: ParentEntryType,
                 data: Iterable[EntryType] = None):
        self.__parent_entry = parent_entry
        self.__data = []
        if data is not None:
            self.__data = [
                self.__parent_entry.relative_pointer(d) for d in data]

    def insert(self, index: int, entry: EntryType):
        self.__data.insert(index, entry.tid)

    def __getitem__(self, index: Union[int, slice]
                    ) -> Union[EntryType, MutableSequence[EntryType]]:
        if isinstance(index, slice):
            return [self.__parent_entry.from_pointer(d) for d in
                    self.__data[index]]
        else:
            return self.__parent_entry.from_pointer(self.__data[index])

    def __setitem__(
            self, index: Union[int, slice],
            value: Union[EntryType, Iterable[EntryType]]) -> None:
        if isinstance(value, Iterable):
            d_value = [v.tid for v in value]
        else:
            d_value = value.tid
        self.__data[index] = d_value

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.__data[index]

    def __len__(self) -> int:
        return len(self.__data)


KeyType = TypeVar('KeyType', bound=Hashable)
ValueType = TypeVar('ValueType', bound=Entry)


class FDict(Generic[KeyType, ValueType], MutableMapping):
    """
    FDict allows the values to be Forte entries. FDict will internally
    stores the entry as their tid to avoid nesting. Note that key is not
    supported to be entries now.
    """

    def __init__(self, parent_entry: ParentEntryType,
                 data: Dict[KeyType, ValueType] = None):
        self.__parent_entry = parent_entry
        self.__data: Dict[KeyType, Any] = {}

        if data is not None:
            self.__data = {
                k: self.__parent_entry.relative_pointer(
                    v) for k, v in data.items()}

    def __setitem__(self, k: KeyType, v: ValueType) -> None:
        self.__data[k] = v.tid

    def __delitem__(self, k: KeyType) -> None:
        del self.__data[k]

    def __getitem__(self, k: KeyType) -> ValueType:
        return self.__parent_entry.from_pointer(self.__data[k])

    def __len__(self) -> int:
        return len(self.__data)

    def __iter__(self) -> Iterator[ValueType]:
        yield from self.__data


class Pointer(BasePointer):
    """
    A pointer that points to an entry in the current pack, this is basically
    containing the entry's tid.
    """

    def __init__(self, tid: int):
        self._tid: int = tid

    @property
    def tid(self):
        return self._tid

    def __str__(self):
        return str(self.tid)


class MpPointer(BasePointer):
    """
    Multi pack Pointer. A pointer that refers to an entry of one of the pack in
    the multi pack. This contains the pack's index and the entries' tid.
    """

    def __init__(self, pack_index: int, tid: int):
        self._pack_index: int = pack_index
        self._tid: int = tid

    @property
    def pack_index(self):
        return self._pack_index

    @property
    def tid(self):
        return self._tid

    def __str__(self):
        return str((self.pack_index, self.tid))


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
        r"""This will set the `parent` of the current instance with given Entry
        The parent is saved internally by its pack specific index key.

        Args:
            parent: The parent entry.
        """
        raise NotImplementedError

    @abstractmethod
    def set_child(self, child: Entry):
        r"""This will set the `child` of the current instance with given Entry
        The child is saved internally by its pack specific index key.

        Args:
            child: The child entry
        """
        raise NotImplementedError

    @abstractmethod
    def get_parent(self) -> Entry:
        r"""Get the parent entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link
             from the given :class:`DataPack`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_child(self) -> Entry:
        r"""Get the child entry of the link.

        Returns:
             An instance of :class:`Entry` that is the child of the link
             from the given :class:`DataPack`.
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
    r"""Group is an entry that represent a group of other entries. For example,
    a "coreference group" is a group of coreferential entities. Each group will
    store a set of members, no duplications allowed.

    This is the :class:`BaseGroup` interface. Specific member constraints are
    defined in the inherited classes.
    """
    MemberType: Type[EntryType]

    def __init__(
            self, pack: ContainerType, members: Optional[Set[EntryType]] = None
    ):
        super().__init__(pack)
        if members is not None:
            self.add_members(members)

    @abstractmethod
    def add_member(self, member: EntryType):
        r"""Add one entry to the group.

        Args:
            member: One member to be added to the group.
        """
        raise NotImplementedError

    def add_members(self, members: Iterable[EntryType]):
        r"""Add members to the group.

        Args:
            members: An iterator of members to be added to the group.
        """
        for member in members:
            self.add_member(member)

    def __hash__(self):
        r"""The hash function of :class:`Group`.

        Users can define their own hash function by themselves but this must
        be consistent to :meth:`eq`.
        """
        return hash((type(self), tuple(self.get_members())))

    def __eq__(self, other):
        r"""The eq function of :class:`Group`. By default, :class:`Group`
        objects are regarded as the same if they have the same type, members,
        and are generated by the same component.

        Users can define their own eq function by themselves but this must
        be consistent to :meth:`hash`.
        """
        if other is None:
            return False
        return (type(self), self.get_members()) == (
            type(other), other.get_members())

    @abstractmethod
    def get_members(self) -> Set[EntryType]:
        r"""Get the member entries in the group.

        Returns:
             A set of instances of :class:`Entry` that are the members of the
             group.
        """
        raise NotImplementedError

    @property
    def index_key(self) -> int:
        return self.tid


GroupType = TypeVar("GroupType", bound=BaseGroup)
LinkType = TypeVar('LinkType', bound=BaseLink)
