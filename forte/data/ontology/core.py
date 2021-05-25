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
import uuid
import warnings
from abc import abstractmethod, ABC
from collections.abc import MutableSequence, MutableMapping
from dataclasses import dataclass
from typing import (
    Iterable, Optional, Type, Hashable, TypeVar, Generic,
    Union, Dict, Iterator, get_type_hints, overload, List)

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
        self._tid: int = uuid.uuid4().int
        self._embedding: np.ndarray = np.empty(0)
        self.pack._validate(self)
        self.pack.on_entry_creation(self)

    def regret_creation(self):
        self.__pack.regret_creation(self)

    def __getstate__(self):
        r"""In serialization, the pack is not serialize, and it will be set
        by the container.

        This also implies that it is not advised to serialize an entry on its
        own, without the ``Container`` as the context, there is little semantics
        remained in an entry.
        """
        state = self.__dict__.copy()
        # During serialization, convert the numpy array as a list.
        emb = list(self._embedding.tolist())
        if len(emb) == 0:
            state.pop("_embedding")
        else:
            state["_embedding"] = emb
        state.pop('_Entry__pack')
        return state

    def __setstate__(self, state):
        # Recover the internal __field_modified dict for the entry.
        # NOTE: the __pack will be set via set_pack from the Pack side.
        # self.__dict__['_Entry__field_modified'] = set()

        # During de-serialization, convert the list back to numpy array.
        if "_embedding" in state:
            state["_embedding"] = np.array(state["_embedding"])
        else:
            state["_embedding"] = np.empty(0)
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
        return self.__pack.pack_id  # type: ignore

    def set_pack(self, pack: ContainerType):
        self.__pack = pack

    def as_pointer(self, from_entry: "Entry"):
        """
        Return this entry as a pointer of this entry relative to the
        ``from_entry``.

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

    def resolve_pointer(self, ptr: BasePointer):
        """
        Resolve into an entry on the provided pointer ``ptr`` from this entry.

        Args:
            ptr:

        Returns:

        """
        if isinstance(ptr, Pointer):
            return self.pack.get_entry(ptr.tid)
        else:
            raise TypeError(
                f"Unsupported pointer type {ptr.__class__} for entry")

    def entry_type(self) -> str:
        """Return the full name of this entry type."""
        module = self.__class__.__module__
        if module is None or module == str.__class__.__module__:
            return self.__class__.__name__
        else:
            return module + '.' + self.__class__.__name__

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
            if key not in hints.keys():
                warnings.warn(
                    f"Base on attributes in entry definition, "
                    f"the [{key}] attribute_name does not exist in the "
                    f"[{type(self).__name__}] that you specified to add to."
                )
            is_valid = check_type(value, hints[key])
            if not is_valid:
                warnings.warn(
                    f"Based on type annotation, "
                    f"the [{key}] attribute of [{type(self).__name__}] "
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
            self.__pack.record_field(self.tid, key)

    def __getattribute__(self, item):
        try:
            v = super().__getattribute__(item)
        except AttributeError:
            # For all unknown attributes, return None.
            return None

        if isinstance(v, BasePointer):
            # Using the pointer to get the entry.
            return self.resolve_pointer(v)
        else:
            return v

    def __eq__(self, other):
        r"""The eq function for :class:`Entry` objects.
        To be implemented in each subclass.
        """
        if other is None:
            return False

        return (type(self), self._tid) == (type(other), other.tid)

    def __lt__(self, other):
        r"""By default, compared based on type string.
        """
        return (str(type(self))) < (str(type(other)))

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
            # Save a pointer of the value.
            self.__dict__[key] = value.as_pointer(self)
        else:
            super().__setattr__(key, value)

    def as_pointer(self, from_entry: "Entry") -> "Pointer":
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

    def resolve_pointer(self, ptr: BasePointer) -> Entry:
        if isinstance(ptr, Pointer):
            return self.pack.get_entry(ptr.tid)
        elif isinstance(ptr, MpPointer):
            return self.pack.packs[ptr.pack_index].get_entry(ptr.tid)
        else:
            raise TypeError(f"Unknown pointer type {ptr.__class__}")


EntryType = TypeVar("EntryType", bound=Entry)

ParentEntryType = TypeVar("ParentEntryType", bound=Entry)


# TODO: Cannot pickle with FList[CorefQuestionAnswers], have generic problems.
class FList(Generic[ParentEntryType], MutableSequence):
    """
    FList allows the elements to be Forte entries. FList will internally
    stores the entry as their tid to avoid nesting.
    """

    def __init__(self, parent_entry: ParentEntryType,
                 data: Optional[Iterable[EntryType]] = None):
        super().__init__()
        self.__parent_entry = parent_entry
        self.__data: List[BasePointer] = []
        if data is not None:
            self.__data = [d.as_pointer(self.__parent_entry) for d in data]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_FList__parent_entry')
        return state

    def insert(self, index: int, entry: EntryType):
        self.__data.insert(index, entry.as_pointer(self.__parent_entry))

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> EntryType:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence:
        ...

    def __getitem__(self, index: Union[int, slice]
                    ) -> Union[EntryType, MutableSequence]:
        if isinstance(index, slice):
            return [self.__parent_entry.resolve_pointer(d) for d in
                    self.__data[index]]
        else:
            return self.__parent_entry.resolve_pointer(self.__data[index])

    def __setitem__(
            self, index: Union[int, slice],
            value: Union[EntryType, Iterable[EntryType]]) -> None:
        # pylint: disable=isinstance-second-argument-not-valid-type
        # TODO: Disable until fix: https://github.com/PyCQA/pylint/issues/3507
        if isinstance(index, int):
            # Assert for mypy: https://github.com/python/mypy/issues/7858
            assert isinstance(value, Entry)
            self.__data[index] = value.as_pointer(self.__parent_entry)
        else:
            assert isinstance(value, Iterable)
            self.__data[index] = [
                v.as_pointer(self.__parent_entry) for v in value]

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
                 data: Optional[Dict[KeyType, ValueType]] = None):
        super().__init__()

        self.__parent_entry = parent_entry
        self.__data: Dict[KeyType, BasePointer] = {}

        if data is not None:
            self.__data = {
                k: v.as_pointer(self.__parent_entry) for k, v in data.items()}

    def __setitem__(self, k: KeyType, v: ValueType) -> None:
        try:
            self.__data[k] = v.as_pointer(self.__parent_entry)
        except AttributeError as e:
            raise AttributeError(
                f"Item of the FDict must be of type entry, "
                f"got {v.__class__}") from e

    def __delitem__(self, k: KeyType) -> None:
        del self.__data[k]

    def __getitem__(self, k: KeyType) -> ValueType:
        return self.__parent_entry.resolve_pointer(self.__data[k])

    def __len__(self) -> int:
        return len(self.__data)

    def __iter__(self) -> Iterator[KeyType]:
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
            self, pack: ContainerType,
            members: Optional[Iterable[EntryType]] = None
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
    def get_members(self) -> List[EntryType]:
        r"""Get the member entries in the group.

        Returns:
             Instances of :class:`Entry` that are the members of the
             group.
        """
        raise NotImplementedError

    @property
    def index_key(self) -> int:
        return self.tid


GroupType = TypeVar("GroupType", bound=BaseGroup)
LinkType = TypeVar('LinkType', bound=BaseLink)
