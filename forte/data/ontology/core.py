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
    Tuple,
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
import math
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
    "Grid",
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


class Grid:
    """
    Regular grid with a grid configuration dependent on the image size.
    It is a data structure used to retrieve grid-related objects such as grid
    cells from the image. Grid itself doesn't store image data but only data
    related to grid configurations such as grid shape and image size.

    Based on the image size and the grid shape,
    we compute the height and the width of grid cells.
    For example, if the image size (image_height,image_width) is (640, 480)
    and the grid shape (height, width) is (2, 3)
    the size of grid cells (self.c_h, self.c_w) will be (320, 160).

    However, when the image size is not divisible by the grid shape, we round
    up the resulting divided size(floating number) to an integer.
    In this way, as each grid cell possibly takes one more pixel,
    we make the last grid cell per column and row
    size(height and width) to be the remainder of the image size divided by the
    grid cell size which is smaller than other grid cell.
    For example, if the image
    size is (128, 128) and the grid shape is (13, 13), the first 12 grid cells
    per column and row will have a size of (10, 10) since 128/13=9.85, so we
    round up to 10. The last grid cell per column and row will have a size of
    (8, 8) since 128%10=8.

    Args:
        height: the number of grid cell per column.
        width: the number of grid cell per row.
        image_height: the number of pixels per column in the image.
        image_width: the number of pixels per row in the image.
    """

    def __init__(
        self,
        height: int,
        width: int,
        image_height: int,
        image_width: int,
    ):
        if image_height <= 0 or image_width <= 0:
            raise ValueError(
                "both image height and width must be positive"
                f"but the image shape is {(image_height, image_width)}"
                "please input a valid image shape"
            )
        if height <= 0 or width <= 0:
            raise ValueError(
                f"height({height}) and "
                f"width({width}) both must be larger than 0"
            )
        if height >= image_height or width >= image_width:
            raise ValueError(
                "Grid height and width must be smaller than image height and width"
            )

        self._height = height
        self._width = width

        # We require each grid to be bounded/intialized with one image size since
        # the number of different image shapes are limited per computer vision task.
        # For example, we can only have one image size (640, 480) from a CV dataset,
        # and we could augment the dataset with few other image sizes
        # (320, 240), (480, 640). Then there are only three image sizes.
        # Therefore, it won't be troublesome to
        # have a grid for each image size, and we can check the image size during the
        # initialization of the grid.

        # By contrast, if we don't initialize it with any
        # image size and pass the image size directly into the method/operation on
        # the fly, the API would be more complex and image size check would be
        # repeated everytime the method is called.
        self._image_height = image_height
        self._image_width = image_width

        # if the resulting size of grid is not an integer, we round it up.
        # The last grid cell per row and column might be out of the image size
        # since we constrain the maximum pixel locations by the image size
        self.c_h, self.c_w = (
            math.ceil(image_height / self._height),
            math.ceil(image_width / self._width),
        )

    def _get_image_within_grid_cell(
        self,
        img_arr: np.ndarray,
        h_idx: int,
        w_idx: int,
    ) -> np.ndarray:
        """
        Get the array data within a grid cell from the image data.
        The array is a masked version of the original image, and it has
        the same size as the original image. The array entries that are not
        within the grid cell will masked as zeros. The image array entries that
        are within the grid cell will kept.
        Note: all indices are zero-based and counted from top left corner of
        the image.

        Args:
            img_arr: image data represented as a numpy array.
            h_idx: the zero-based height(row) index of the grid cell in the
                grid, the unit is one grid cell.
            w_idx: the zero-based width(column) index of the grid cell in the
                grid, the unit is one grid cell.

        Raises:
            ValueError: ``h_idx`` is out of the range specified by ``height``.
            ValueError: ``w_idx`` is out of the range specified by ``width``.

        Returns:
            numpy array that represents the grid cell.
        """
        if not 0 <= h_idx < self._height:
            raise ValueError(
                f"input parameter h_idx ({h_idx}) is"
                "out of scope of h_idx range"
                f" {(0, self._height)}"
            )
        if not 0 <= w_idx < self._width:
            raise ValueError(
                f"input parameter w_idx ({w_idx}) is"
                "out of scope of w_idx range"
                f" {(0, self._width)}"
            )

        return img_arr[
            h_idx * self.c_h : min((h_idx + 1) * self.c_h, self._image_height),
            w_idx * self.c_w : min((w_idx + 1) * self.c_w, self._image_width),
        ]

    def get_overlapped_grid_cell_indices(
        self, image_arr: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Get the grid cell indices in the form of (height index, width index)
        that image array overlaps with.

        Args:
            image_arr: image data represented as a numpy array.

        Returns:
            a list of tuples that represents the grid cell indices that image array overlaps with.
        """
        grid_cell_indices = []
        for h_idx in range(self._height):
            for w_idx in range(self._width):
                if (
                    np.sum(
                        self._get_image_within_grid_cell(
                            image_arr, h_idx, w_idx
                        )
                    )
                    > 0
                ):
                    grid_cell_indices.append((h_idx, w_idx))
        return grid_cell_indices

    def get_grid_cell_center(self, h_idx: int, w_idx: int) -> Tuple[int, int]:
        """
        Get the center pixel position of the grid cell at the specific height
        index and width index in the ``Grid``.
        The computation of the center position of the grid cell is
        dividing the grid cell height range (unit: pixel) and
        width range (unit: pixel) by 2 (round down)
        Suppose an edge case that a grid cell has a height range
        (unit: pixel) of (0, 3)
        and a width range (unit: pixel) of (0, 3) the grid cell center
        would be (1, 1).
        Since the grid cell size is usually very large,
        the offset of the grid cell center is minor.
        Note: all indices are zero-based and counted from top left corner of
        the grid.
        Args:
            h_idx: the height(row) index of the grid cell in the grid,
                the unit is one grid cell.
            w_idx: the width(column) index of the grid cell in the
                grid, the unit is one grid cell.
        Returns:
            A tuple of (y index, x index)
        """

        return (
            (h_idx * self.c_h + min((h_idx + 1) * self.c_h, self._image_height))
            // 2,
            (w_idx * self.c_w + min((w_idx + 1) * self.c_w, self._image_width))
            // 2,
        )

    @property
    def num_grid_cells(self):
        return self._height * self._width

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def __repr__(self):
        return str(
            (self._height, self._width, self._image_height, self._image_width)
        )

    def __eq__(self, other):
        if other is None:
            return False
        return (
            self._height,
            self._width,
            self._image_height,
            self._image_width,
        ) == (
            other._height,
            other._width,
            other.image_height,
            other.image_width,
        )

    def __hash__(self):
        return hash(
            (self._height, self._width, self._image_height, self._image_width)
        )


GroupType = TypeVar("GroupType", bound=BaseGroup)
LinkType = TypeVar("LinkType", bound=BaseLink)

ENTRY_TYPE_DATA_STRUCTURES = (FDict, FList)
