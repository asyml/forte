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

from abc import abstractmethod, ABC
from collections.abc import MutableSequence, MutableMapping
from dataclasses import dataclass
from typing import (
    Iterable,
    Optional,
    Type,
    Hashable,
    TypeVar,
    Generic,
    Union,
    Dict,
    Iterator,
    overload,
    List,
)

import numpy as np

from forte.data.container import ContainerType

__all__ = [
    "Entry",
    "BaseLink",
    "BaseGroup",
    "LinkType",
    "GroupType",
    "EntryType",
    "FDict",
    "FList",
    "FNdArray",
    "MultiEntry",
]

default_entry_fields = [
    "_Entry__pack",
    "_tid",
    "_embedding",
    "_span",
    "_begin",
    "_end",
    "_parent",
    "_child",
    "_members",
    "_Entry__field_modified",
    "field_records",
    "creation_records",
    "_id_manager",
]

unserializable_fields = [
    # This may be related to typing, but cannot be supported by typing.
    "__orig_class__",
]


@dataclass
class Entry(Generic[ContainerType]):
    r"""The base class inherited by all NLP entries. This is the main data type
    for all in-text NLP analysis results. The main sub-types are
    :class:`~forte.data.ontology.top.Annotation`, ``Link``, ``Generics``, and
    ``Group``.

    An :class:`forte.data.ontology.top.Annotation` object represents a
    span in text.

    A :class:`forte.data.ontology.top.Link` object represents a binary
    link relation between two entries.

    A :class:`forte.data.ontology.top.Generics` object.

    A :class:`forte.data.ontology.top.Group` object represents a
    collection of multiple entries.

    Main Attributes:

        - embedding: The embedding vectors (numpy array of floats) of this
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

    # using property decorator
    # a getter function for self._embedding
    @property
    def embedding(self):
        r"""Get the embedding vectors (numpy array of floats) of the entry."""
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
            id of the entry
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
            id of the pack that contains this entry.
        """
        return self.__pack.pack_id  # type: ignore

    def set_pack(self, pack: ContainerType):
        self.__pack = pack

    def entry_type(self) -> str:
        """Return the full name of this entry type."""
        module = self.__class__.__module__
        if module is None or module == str.__class__.__module__:
            return self.__class__.__name__
        else:
            return module + "." + self.__class__.__name__

    # def _check_attr_type(self, key, value):
    #     """
    #     Use the type hint to validate whether the provided value is as expected.
    #
    #     Args:
    #         key:  The field name.
    #         value: The field value.
    #
    #     Returns:
    #
    #     """
    #     if key not in default_entry_fields:
    #         hints = get_type_hints(self.__class__)
    #         if key not in hints.keys():
    #             warnings.warn(
    #                 f"Base on attributes in entry definition, "
    #                 f"the [{key}] attribute_name does not exist in the "
    #                 f"[{type(self).__name__}] that you specified to add to."
    #             )
    #         is_valid = check_type(value, hints[key])
    #         if not is_valid:
    #             warnings.warn(
    #                 f"Based on type annotation, "
    #                 f"the [{key}] attribute of [{type(self).__name__}] "
    #                 f"should be [{hints[key]}], but got [{type(value)}]."
    #             )

    # def __setattr__(self, key, value):
    #     super().__setattr__(key, value)
    #
    #     if isinstance(value, Entry):
    #         if value.pack == self.pack:
    #             # Save a pointer to the value from this entry.
    #             self.__dict__[key] = Pointer(value.tid)
    #         else:
    #             raise PackDataException(
    #                 "An entry cannot refer to entries in another data pack."
    #             )
    #     else:
    #         super().__setattr__(key, value)
    #
    #     # We add the record to the system.
    #     if key not in default_entry_fields:
    #         self.__pack.record_field(self.tid, key)

    # def __getattribute__(self, item):
    #     try:
    #         v = super().__getattribute__(item)
    #     except AttributeError:
    #         # For all unknown attributes, return None.
    #         return None
    #
    #     if isinstance(v, BasePointer):
    #         # Using the pointer to get the entry.
    #         return self._resolve_pointer(v)
    #     else:
    #         return v

    def __eq__(self, other):
        r"""
        The eq function for :class:`~forte.data.ontology.core.Entry` objects.
        Can be further implemented in each subclass.

        Args:
            other:

        Returns:
            None
        """
        if other is None:
            return False

        return (type(self), self._tid) == (type(other), other.tid)

    def __lt__(self, other):
        r"""By default, compared based on type string."""
        return (str(type(self))) < (str(type(other)))

    def __hash__(self) -> int:
        r"""The hash function for :class:`~forte.data.ontology.core.Entry` objects.
        To be implemented in each subclass.
        """
        return hash((type(self), self._tid))

    @property
    def index_key(self) -> Hashable:
        # Think about how to use the index key carefully.
        return self._tid


class MultiEntry(Entry, ABC):
    r"""The base class for multi-pack entries. The main sub-types are
    ``MultiPackLink``, ``MultiPackGenerics``, and ``MultiPackGroup``.

    A :class:`forte.data.ontology.top.MultiPackLink` object represents a binary
    link relation between two entries between different data packs.

    A :class:`forte.data.ontology.top.MultiPackGroup` object represents a
    collection of multiple entries among different data packs.
    """
    pass


EntryType = TypeVar("EntryType", bound=Entry)

ParentEntryType = TypeVar("ParentEntryType", bound=Entry)


class FList(Generic[ParentEntryType], MutableSequence):
    """
    FList allows the elements to be Forte entries. FList will internally
    deal with a refernce list from DataStore which stores the entry as their
    tid to avoid nesting.
    """

    def __init__(
        self,
        parent_entry: ParentEntryType,
        data: Optional[List[int]] = None,
    ):
        super().__init__()
        self.__parent_entry = parent_entry
        self.__data: List[int] = [] if data is None else data

    def __eq__(self, other):
        return self.__data == other._FList__data

    def __getstate__(self):
        state = self.__dict__.copy()
        # Parent entry cannot be serialized, should be set by the parent with
        #  _set_parent.
        state.pop("_FList__parent_entry")

        state["data"] = state.pop("_FList__data")

        # We make a copy of the whole state but there are items cannot be
        # serialized.
        for f in unserializable_fields:
            if f in state:
                state.pop(f)
        return state

    def __setstate__(self, state):
        state["_FList__data"] = state.pop("data")
        self.__dict__.update(state)

    def _set_parent(self, parent_entry: ParentEntryType):
        self.__parent_entry = parent_entry

    def insert(self, index: int, entry: EntryType):
        self.__data.insert(index, entry.tid)

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> EntryType:
        ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence:
        ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[EntryType, MutableSequence]:
        if isinstance(index, slice):
            return [
                self.__parent_entry.pack.get_entry(tid)
                for tid in self.__data[index]
            ]
        else:
            return self.__parent_entry.pack.get_entry(self.__data[index])

    def __setitem__(
        self,
        index: Union[int, slice],
        value: Union[EntryType, Iterable[EntryType]],
    ) -> None:
        if isinstance(index, int):
            self.__data[index] = value.tid  # type: ignore
        else:
            self.__data[index] = [v.tid for v in value]  # type: ignore

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.__data[index]

    def __len__(self) -> int:
        return len(self.__data)


KeyType = TypeVar("KeyType", bound=Hashable)
ValueType = TypeVar("ValueType", bound=Entry)


class FDict(Generic[KeyType, ValueType], MutableMapping):
    """
    FDict allows the values to be Forte entries. FDict will internally
    deal with a refernce dict from DataStore which stores the entry as their
    tid to avoid nesting. Note that key is not supported to be entries now.
    """

    def __init__(
        self,
        parent_entry: ParentEntryType,
        data: Optional[Dict[KeyType, int]] = None,
    ):
        super().__init__()

        self.__parent_entry = parent_entry
        self.__data: Dict[KeyType, int] = {} if data is None else data

    def _set_parent(self, parent_entry: ParentEntryType):
        self.__parent_entry = parent_entry

    def __eq__(self, other):
        return self.__data == other._FDict__data

    def __getstate__(self):
        state = self.__dict__.copy()
        # The __parent_entry need to be assigned via its parent entry,
        # so a serialized dict may not have the following key ready sometimes.
        state.pop("_FDict__parent_entry")

        state["data"] = state.pop("_FDict__data")

        # We make a copy of the whole state but there are items cannot be
        # serialized.
        for f in unserializable_fields:
            if f in state:
                state.pop(f)
        return state

    def __setstate__(self, state):
        state["_FDict__data"] = state.pop("data")
        self.__dict__.update(state)

    def __setitem__(self, k: KeyType, v: ValueType) -> None:
        try:
            self.__data[k] = v.tid
        except AttributeError as e:
            raise AttributeError(
                f"Item of the FDict must be of type entry, "
                f"got {v.__class__}"
            ) from e

    def __delitem__(self, k: KeyType) -> None:
        del self.__data[k]

    def __getitem__(self, k: KeyType) -> ValueType:
        return self.__parent_entry.pack.get_entry(self.__data[k])

    def __len__(self) -> int:
        return len(self.__data)

    def __iter__(self) -> Iterator[KeyType]:
        yield from self.__data


class FNdArray:
    """
    FNdArray is a wrapper of a NumPy array that stores shape and data type
    of the array if they are specified. Only when both shape and data type
    are provided, will FNdArray initialize a placeholder array through
    np.ndarray(shape, dtype=dtype).
    More details about np.ndarray(...):
    https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
    """

    def __init__(
        self, dtype: Optional[str] = None, shape: Optional[List[int]] = None
    ):
        super().__init__()
        self._dtype: Optional[np.dtype] = (
            np.dtype(dtype) if dtype is not None else dtype
        )
        self._shape: Optional[tuple] = (
            tuple(shape) if shape is not None else shape
        )
        self._data: Optional[np.ndarray] = None
        if dtype and shape:
            self._data = np.ndarray(shape, dtype=dtype)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, array: Union[np.ndarray, List]):
        if isinstance(array, np.ndarray):
            if self.dtype and not np.issubdtype(array.dtype, self.dtype):
                raise TypeError(
                    f"Expecting type or subtype of {self.dtype}, but got {array.dtype}."
                )
            if self.shape and self.shape != array.shape:
                raise AttributeError(
                    f"Expecting shape {self.shape}, but got {array.shape}."
                )
            self._data = array

        elif isinstance(array, list):
            array_np = np.array(array, dtype=self.dtype)
            if self.shape and self.shape != array_np.shape:
                raise AttributeError(
                    f"Expecting shape {self.shape}, but got {array_np.shape}."
                )
            self._data = array_np

        else:
            raise ValueError(
                f"Can only accept numpy array or python list, but got {type(array)}"
            )

        # Stored dtype and shape should match to the provided array's.
        self._dtype = self._data.dtype
        self._shape = self._data.shape


class BaseLink(Entry, ABC):
    def __init__(
        self,
        pack: ContainerType,
        parent: Optional[Entry] = None,
        child: Optional[Entry] = None,
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
             An instance of :class:`~forte.data.ontology.core.Entry` that is the child of the link
             from the given :class:`~forte.data.data_pack.DataPack`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_child(self) -> Entry:
        r"""Get the child entry of the link.

        Returns:
             An instance of :class:`~forte.data.ontology.core.Entry` that is the child of the link
             from the given :class:`~forte.data.data_pack.DataPack`.
        """
        raise NotImplementedError

    def __eq__(self, other):
        if other is None:
            return False
        return (type(self), self.get_parent(), self.get_child()) == (
            type(other),
            other.get_parent(),
            other.get_child(),
        )

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
        self, pack: ContainerType, members: Optional[Iterable[EntryType]] = None
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
            type(other),
            other.get_members(),
        )

    @abstractmethod
    def get_members(self) -> List[EntryType]:
        r"""Get the member entries in the group.

        Returns:
             Instances of :class:`~forte.data.ontology.core.Entry` that are the members of the
             group.
        """
        raise NotImplementedError

    @property
    def index_key(self) -> int:
        return self.tid


GroupType = TypeVar("GroupType", bound=BaseGroup)
LinkType = TypeVar("LinkType", bound=BaseLink)
