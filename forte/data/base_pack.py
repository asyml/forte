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

import logging
import copy
import gzip
import json
import pickle
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import (
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    Iterator,
    Dict,
    Any,
    Iterable,
)
from functools import partial
from inspect import isclass
from typing_inspect import is_forward_ref
from packaging.version import Version
import jsonpickle

from forte.common import ProcessExecutionException, EntryNotFoundError
from forte.common.constants import JSON_CLASS_FIELD, JSON_STATE_FIELD
from forte.data.index import BaseIndex
from forte.data.base_store import BaseStore
from forte.data.container import EntryContainer
from forte.data.ontology.core import (
    Entry,
    EntryType,
    GroupType,
    LinkType,
    FList,
    FDict,
    FNdArray,
    ENTRY_TYPE_DATA_STRUCTURES,
)
from forte.version import (
    PACK_VERSION,
    DEFAULT_PACK_VERSION,
    PACK_ID_COMPATIBLE_VERSION,
)
from forte.utils import get_full_module_name, get_class

logger = logging.getLogger(__name__)

__all__ = ["BasePack", "BaseMeta", "PackType"]


class BaseMeta:
    r"""Basic Meta information for both :class:`~forte.data.data_pack.DataPack`
    and :class:`~forte.data.multi_pack.MultiPack`.

    Args:
        pack_name:  An name to identify the data pack, which is helpful in
           situation like serialization. It is suggested that the packs should
           have different doc ids.
    Attributes:
        record: Initialized as a dictionary. This is not a required field.
            The key of the record should be the entry type and values should
            be attributes of the entry type. All the information would be used
            for consistency checking purpose if the pipeline is initialized with
            `enforce_consistency=True`.
    """

    def __init__(self, pack_name: Optional[str] = None):
        self.pack_name: Optional[str] = pack_name
        self._pack_id: int = uuid.uuid4().int
        self.record: Dict[str, Set[str]] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Re-obtain the pack manager during deserialization.
        Args:
            state:

        Returns:

        """
        self.__dict__.update(state)

    @property
    def pack_id(self) -> int:
        return self._pack_id


class BasePack(EntryContainer[EntryType, LinkType, GroupType]):
    r"""The base class of :class:`~forte.data.data_pack.DataPack` and
    :class:`~forte.data.multi_pack.MultiPack`.

    Args:
        pack_name: a string name of the pack.

    """

    # pylint: disable=too-many-public-methods
    def __init__(self, pack_name: Optional[str] = None):
        super().__init__()
        self.pack_version: str = PACK_VERSION

        self._meta: BaseMeta = self._init_meta(pack_name)
        self._index: BaseIndex = BaseIndex()

        self._data_store: BaseStore

        self.__control_component: Optional[str] = None

        # This Dict maintains a mapping from entry's tid to the component
        # name associated with the entry.
        # The component name is used for tracking the "creator" of this entry.
        self._pending_entries: Dict[int, Optional[str]] = {}

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_index")
        state.pop("_pending_entries")
        state.pop("_BasePack__control_component")
        return state

    def __setstate__(self, state):
        # Pack version checking. We will no longer provide support for
        # serialized Pack whose "pack_version" is less than
        # PACK_ID_COMPATIBLE_VERSION.
        pack_version: str = (
            state["pack_version"]
            if "pack_version" in state
            else DEFAULT_PACK_VERSION
        )
        if Version(pack_version) < Version(PACK_ID_COMPATIBLE_VERSION):
            raise ValueError(
                "The pack cannot be deserialized because its version "
                f"{pack_version} is outdated. We only support pack with "
                f"version greater or equal to {PACK_ID_COMPATIBLE_VERSION}"
            )
        super().__setstate__(state)
        if "meta" in self.__dict__:
            self._meta = self.__dict__.pop("meta")
        self.__control_component = None
        self._pending_entries = {}

    @abstractmethod
    def _init_meta(self, pack_name: Optional[str] = None) -> BaseMeta:
        raise NotImplementedError

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self._meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self._meta, k, v)

    @property
    def pack_id(self):
        return self._meta.pack_id

    @abstractmethod
    def __iter__(self) -> Iterator[EntryType]:
        raise NotImplementedError

    def __del__(self):
        if len(self._pending_entries) > 0:
            raise ProcessExecutionException(
                f"There are {len(self._pending_entries)} "
                f"entries not added to the index correctly."
            )

    @property
    def pack_name(self):
        return self._meta.pack_name

    @pack_name.setter
    def pack_name(self, pack_name: str):
        """
        Update the pack name of this pack.

        Args:
            pack_name: The new doc id.

        Returns:

        """
        self._meta.pack_name = pack_name

    @classmethod
    def _deserialize(
        cls,
        data_source: Union[Path, str],
        serialize_method: str = "json",
        zip_pack: bool = False,
    ) -> "BasePack[Any, Any, Any]":
        """
        This function should deserialize a Pack from a string. The
        implementation should decide the specific pack type.

        Args:
            data_source: The data path containing pack data. The content
              of the data could be string or bytes depending on the method of
              serialization.
            serialize_method: The method used to serialize the data, this
              should be the same as how serialization is done. The current
              options are `json`, `jsonpickle` and `pickle`. The default method
              is `json`.
            zip_pack: Boolean value indicating whether the input source is
              zipped.

        Returns:
            An pack object deserialized from the data.
        """
        _open = gzip.open if zip_pack else open

        if serialize_method in ("jsonpickle", "json"):
            with _open(data_source, mode="rt") as f:  # type: ignore
                pack = cls.from_string(f.read(), json_method=serialize_method)
        else:
            with _open(data_source, mode="rb") as f:  # type: ignore
                pack = pickle.load(f)

            if not hasattr(pack, "pack_version"):
                pack.pack_version = DEFAULT_PACK_VERSION

        return pack

    @classmethod
    def from_string(
        cls, data_content: str, json_method: str = "json"
    ) -> "BasePack":
        if json_method == "jsonpickle":
            pack = jsonpickle.decode(data_content)
        elif json_method == "json":

            def object_hook(json_dict):
                """
                Custom object hook for JSON deserialization. It will call
                `__setstate__` to deserialize the json content into a class
                object.
                """
                if json_dict.keys() == {JSON_STATE_FIELD, JSON_CLASS_FIELD}:
                    state = json_dict[JSON_STATE_FIELD]
                    obj_type = get_class(json_dict[JSON_CLASS_FIELD])
                    obj = obj_type.__new__(obj_type)
                    obj.__setstate__(state)
                    return obj
                return json_dict

            pack = json.loads(data_content, object_hook=object_hook)
        else:
            raise ValueError(f"Unsupported JSON method {json_method}.")
        if not hasattr(pack, "pack_version"):
            pack.pack_version = DEFAULT_PACK_VERSION

        return pack

    def delete_entry(self, entry: EntryType):
        r"""Remove the entry from the pack.

        Args:
            entry: The entry to be removed.

        Returns:
            None
        """
        self._data_store.delete_entry(tid=entry.tid)

        # update basic index
        self._index.remove_entry(entry)

        # set other index invalid
        self._index.turn_link_index_switch(on=False)
        self._index.turn_group_index_switch(on=False)

    def add_entry(
        self, entry: Union[Entry, int], component_name: Optional[str] = None
    ) -> EntryType:
        r"""Add an :class:`~forte.data.ontology.core.Entry` object to the
        :class:`~forte.data.base_pack.BasePack` object. Allow duplicate entries in a pack.

        Args:
            entry: An :class:`~forte.data.ontology.core.Entry`
                object to be added to the pack.
            component_name: A name to record that the entry is created by
             this component.

        Returns:
            The input entry itself
        """
        # When added to the pack, make a record.
        self.record_entry(entry, component_name)
        # TODO: Returning the entry itself may not be helpful.
        return self._add_entry(entry)

    @abstractmethod
    def _add_entry(self, entry: Union[Entry, int]) -> EntryType:
        r"""Add an :class:`~forte.data.ontology.core.Entry` object to the
        :class:`~forte.data.base_pack.BasePack` object. Allow duplicate entries in a pack.

        Args:
            entry: An :class:`~forte.data.ontology.core.Entry`
                object to be added to the pack.

        Returns:
            The input entry itself
        """
        raise NotImplementedError

    def add_all_remaining_entries(self, component: Optional[str] = None):
        """
        Calling this function will add the entries that are not added to the
        pack manually.

        Args:
            component: Overwrite the component record with this.

        Returns:
            None
        """
        for entry, c in list(self._pending_entries.items()):
            c_ = component if component else c
            self.add_entry(entry, c_)
        self._pending_entries.clear()

    def to_string(
        self,
        drop_record: Optional[bool] = False,
        json_method: str = "json",
        indent: Optional[int] = None,
    ) -> str:
        """
        Return the string representation (json encoded) of this method.

        Args:
            drop_record: Whether to drop the creation records, default is False.
            json_method: What method is used to convert data pack to json.
              Only supports `json` and `jsonpickle` for now. Default value is
              `json`.
            indent: The indent used for json string.

        Returns: String representation of the data pack.
        """
        if drop_record:
            self._creation_records.clear()
            self._field_records.clear()
        if json_method == "jsonpickle":
            return jsonpickle.encode(self, unpicklable=True, indent=indent)
        elif json_method == "json":

            def json_serialize_handler(obj):
                """
                Custom object handler for JSON serialization. It will call
                `__getstate__` to serialize a class object into the its json
                format.
                """
                if hasattr(obj, "__getstate__"):
                    return {
                        JSON_CLASS_FIELD: get_full_module_name(obj),
                        JSON_STATE_FIELD: obj.__getstate__(),
                    }
                raise TypeError(f"Type {type(obj)} not serializable")

            return json.dumps(
                self, indent=indent, default=json_serialize_handler
            )
        else:
            raise ValueError(f"Unsupported JSON method {json_method}.")

    def serialize(
        self,
        output_path: Union[str, Path],
        zip_pack: bool = False,
        drop_record: bool = False,
        serialize_method: str = "json",
        indent: Optional[int] = None,
    ):
        r"""
        Serializes the data pack to the provided path. The output of this
        function depends on the serialization method chosen.

        Args:
            output_path: The path to write data to.
            zip_pack: Whether to compress the result with `gzip`.
            drop_record: Whether to drop the creation records, default is False.
            serialize_method: The method used to serialize the data. Currently
              supports `json` (outputs str), `jsonpickle` (outputs str) and
              Python's built-in `pickle` (outputs bytes).
            indent: Whether to indent the file if written as JSON.

        Returns: Results of serialization.
        """
        if zip_pack:
            _open = gzip.open
        else:
            _open = open  # type:ignore

        if drop_record:
            self._creation_records.clear()
            self._field_records.clear()

        if serialize_method == "pickle":
            with _open(output_path, mode="wb") as pickle_out:
                pickle.dump(self, pickle_out)
        elif serialize_method in ("jsonpickle", "json"):
            with _open(output_path, mode="wt", encoding="utf-8") as json_out:
                json_out.write(
                    self.to_string(
                        drop_record, json_method=serialize_method, indent=indent
                    )
                )
        else:
            raise NotImplementedError(
                f"Unsupported serialization method {serialize_method}"
            )

    def view(self):
        return copy.deepcopy(self)

    def set_control_component(self, component: str):
        """
        Record the current component that is taking control of this pack.

        Args:
            component: The component that is going to take control

        Returns:

        """
        self.__control_component = component

    def record_entry(
        self, entry: Union[Entry, int], component_name: Optional[str] = None
    ):
        c = component_name

        if c is None:
            # Use the auto-inferred control component.
            c = self.__control_component

        if c is not None:
            tid: int = entry.tid if isinstance(entry, Entry) else entry
            try:
                self._creation_records[c].add(tid)
            except KeyError:
                self._creation_records[c] = {tid}

    def record_field(self, entry_id: int, field_name: str):
        """
        Record who modifies the entry, will be called
        in :class:`~forte.data.ontology.core.Entry`

        Args:
            entry_id: The id of the entry.
            field_name: The name of the field modified.

        Returns:

        """
        c = self.__control_component

        if c is not None:
            try:
                self._field_records[c].add((entry_id, field_name))
            except KeyError:
                self._field_records[c] = {(entry_id, field_name)}

    def on_entry_creation(
        self,
        entry: Entry,
        component_name: Optional[str] = None,
    ):
        """
        Call this when adding a new entry, will be called
        in :class:`~forte.data.ontology.core.Entry` when
        its `__init__` function is called. This method does
        the following 2 operations with regards to creating
        a new entry.

        - All ``dataclass`` attributes of the entry to be created
            are stored in the class level dictionary of
            :class:`~forte.data.ontology.core.Entry` called
            ``cached_attributes_data``. This is used to initialize
            the corresponding entry's objects data store entry
        - On creation of the data store entry, this methods associates
            ``getter`` and ``setter`` properties to all `dataclass`
            attributes of this entry to allow direct interaction
            between the attributes of the entry and their copy being
            stored in the data store. For example, the `setter` method
            updates the data store value of an attribute of a given entry
            whenever the attribute in the entry's object is updated.

        Args:
            entry: The entry to be added.
            component_name: A name to record that the entry is created by
             this component.

        Returns:

        """
        c = component_name

        if c is None:
            # Use the auto-inferred control component.
            c = self.__control_component

        def entry_getter(cls: Entry, attr_name: str):
            """A getter function for dataclass fields of entry object.
            Depending on the value stored in the data store and the type
            of the attribute, the method decides how to process the value.

            - Attributes represented as ``FList`` and ``FDict`` objects are stored
                as list and dictionary respectively in the data store entry. These
                values are converted to ``FList`` and ``FDict`` objects on the fly.
            - When the field contains ``tid``s, we will convert them to entry
                object on the fly. This is done by checking the type
                information of the attribute in the entry object. If the
                attribute is of type ``Entry`` or a ``ForwardRef``, we can
                assume that that value stored in the data store entry represents
                the entry's ``tid``.
            - When values are stored as a tuple, we assume the value represents
                a `subentry` stored in a `MultiPack`.
            - In all other cases, the values are returned in the forms that they
                are stored in the data store entry.

            Args:
                cls: An ``Entry`` class object.
                attr_name: The name of the attribute.

            Returns:
                The value of the required attribute in the form specified
                by the corresponding ``Entry`` class object.
            """

            data_store_ref = (
                cls.pack._data_store  # pylint: disable=protected-access
            )
            attr_val = data_store_ref.get_attribute(
                tid=cls.tid, attr_name=attr_name
            )
            attr_type = data_store_ref.get_attr_type(
                cls.entry_type(), attr_name
            )

            if attr_type[0] in ENTRY_TYPE_DATA_STRUCTURES:
                # Generate FList/FDict object on the fly
                return attr_type[0](parent_entry=cls, data=attr_val)
            elif attr_type == (type(None), (FNdArray,)):
                # Generate FNdArray object on the fly
                fndarray: FNdArray = FNdArray()
                fndarray.set_data_ref(attr_val)
                return fndarray

            # Check dataclass attribute value type
            # If the attribute was an Entry object, only its tid
            # is stored in the DataStore and hence its needs to be converted.

            # Entry objects are stored in data stores by their tid (which is
            # of type int). Thus, if we encounter an int value, we check the
            # type information which is stored as a tuple. if any entry in this
            # tuple is a subclass of Entry or is a ForwardRef to another entry,
            # we can infer that this int value represents the tid of an Entry
            # object and thus must be converted to an object using get_entry
            # before returning.
            if attr_type[1] and any(
                issubclass(entry, Entry)
                if isclass(entry)
                else is_forward_ref(entry)
                for entry in list(attr_type[1])
            ):
                try:
                    if isinstance(attr_val, int):
                        return cls.pack.get_entry(tid=attr_val)

                    # The condition below is to check whether the attribute's value
                    # is a pair of integers - `(pack_id, tid)`. If so we may have
                    # encountered a `tid` that can only be resolved by
                    # `MultiPack.get_subentry`.
                    elif (
                        isinstance(attr_val, (tuple, list))
                        and len(attr_val) == 2
                        and all(
                            isinstance(element, int) for element in attr_val
                        )
                        and hasattr(cls.pack, "get_subentry")
                    ):
                        # Multi pack entry
                        return cls.pack.get_subentry(*attr_val)
                except KeyError:
                    pass
            return attr_val

        def entry_setter(cls: Entry, value: Any, attr_name: str):
            """A setter function for dataclass fields of entry object.
            When the value contains entry objects, we will convert them into
            ``tid``s before storing to ``DataStore``. Additionally, if the entry
            setter method is called on an attribute that does not have a pack
            associated with it (as is the case during initialization), the value
            of the attribute is stored in the class level cache of the ``Entry``
            class. On the other hand, if a pack is associated with the entry,
            the value will directly be stored in the data store.

            Args:
                cls: An ``Entry`` class object.
                value: The value to be assigned to the attribute.
                attr_name: The name of the attribute.
            """
            attr_value: Any

            try:
                pack = cls.pack
            except AttributeError as err:
                # This is the case when an object of an entry that has already been
                # created before (which means an setter and getter properties are
                # associated with its dataclass fields) is trying to be initialized.
                # In this case, a pack is not yet associated with this entry. Thus,
                # we store the initial values dataclass fields of such entries in the
                # _cached_attribute_data of the Entry class.

                # pylint: disable=protected-access
                if cls.entry_type() not in Entry._cached_attribute_data:
                    Entry._cached_attribute_data[cls.entry_type()] = {}

                if (
                    attr_name
                    not in Entry._cached_attribute_data[cls.entry_type()]
                ):
                    Entry._cached_attribute_data[cls.entry_type()][
                        attr_name
                    ] = value
                    return
                else:
                    raise KeyError(
                        "You are trying to overwrite the value "
                        f"of {attr_name} for a data store entry "
                        "before it is created."
                    ) from err

            data_store_ref = (
                pack._data_store  # pylint: disable=protected-access
            )

            attr_type = data_store_ref.get_attr_type(
                cls.entry_type(), attr_name
            )
            # Assumption: Users will not assign value to a FList/FDict field.
            # Only internal methods can set the FList/FDict field, and value's
            # type has to be Iterator[Entry]/Dict[Any, Entry].
            if attr_type[0] is FList:
                try:
                    attr_value = [entry.tid for entry in value]
                except AttributeError as e:
                    raise ValueError(
                        "You are trying to assign value to a `FList` field, "
                        "which can only accept an iterator of `Entry` objects."
                    ) from e
            elif attr_type[0] is FDict:
                try:
                    attr_value = {
                        key: entry.tid for key, entry in value.items()
                    }
                except AttributeError as e:
                    raise ValueError(
                        "You are trying to assign value to a `FDict` field, "
                        "which can only accept a mapping whose values are "
                        "`Entry` objects."
                    ) from e
            elif attr_type == (type(None), (FNdArray,)):
                attr_value = [
                    None if value.dtype is None else value.dtype.str,
                    value.shape,
                    None if value.data is None else value.data.tolist(),
                ]
            elif isinstance(value, Entry):
                attr_value = (
                    value.tid
                    if value.pack.pack_id == cls.pack.pack_id
                    # When value's pack and cls' pack are not the same, we
                    # assume that cls.pack is a MultiPack, which will resolve
                    # `value.tid` using `MultiPack.get_subentry(pack_id, tid)`.
                    # In this case, both pack_id and tid should be stored.
                    else (value.pack.pack_id, value.tid)
                )
            else:
                attr_value = value
            data_store_ref.set_attribute(
                tid=cls.tid, attr_name=attr_name, attr_value=attr_value
            )

        # If this is the first time an entry of this type is
        # created, its attributes do not have a getter and setter
        # property associated with them. We can thus assume that there
        # no key in the _cached_attribute_data dictionary that has yet
        # been created to store the dataclass fields of this entry. Thus,
        # we create an empty dictionary to store the dataclass fields
        # of this new entry and manually add all dataclass attributes
        # that have been initialized to the _cached_attribute_data dict.
        # We fetch the values of all dataclass fields by using the getattr
        # method.
        #
        # Additional note added 2022/01/10:
        # There is a case the above-mentioned implementation won't work
        #
        # When registering functions for payloads of the same hierarchy, exceptions
        # will be thrown, this can be tested using the `payload_decorator_test.py`
        # The reason is roughly because of the following steps:
        #
        #   1. since a parent payload class is registered, we will pass the
        #      `not in Entry._cached_attribute_data` condition. however, since the
        #      child payload inherit the parent payload properties and functions,
        #      it will make some actual `getattr` call
        #   2. For the `getattr` to be successful, this entry needs to have the
        #      tid stored in data store first
        #   3. we have _save_entry_to_data_store call later, which saves the tid, but
        #      it happens later in the code.
        #   4. we also cannot move _save_entry_to_data_store earlier, since the entry
        #       attribute name/value pairs need to be filled in first
        #   Thus this causes a conflict.
        #   Currently, I used a very simple solution that surround the `getattr` call
        #      with a try/except block, this will make the above-mentioned child
        #      routine to pretend it doesn't have any `property`. In reality, it actually
        #      has some `property` registered by the parent, but since the `getattr` failed
        #      we get `None` and pretend it doesn't.

        # pylint: disable=protected-access
        if entry.entry_type() not in Entry._cached_attribute_data:
            Entry._cached_attribute_data[entry.entry_type()] = {}
            for name in entry.__dataclass_fields__:
                try:
                    attr_val = getattr(entry, name, None)
                except KeyError:
                    attr_val = None

                if attr_val is not None:
                    Entry._cached_attribute_data[entry.entry_type()][
                        name
                    ] = attr_val

        # Save the input entry object in DataStore
        self._save_entry_to_data_store(entry=entry)

        # Register property functions for all dataclass fields.
        for name in entry.__dataclass_fields__:
            # Convert the typing annotation to the original class.
            # This will be used to determine if a field is FList/FDict.
            setattr(
                type(entry),
                name,
                # property(fget, fset) will register a conversion layer
                # that specifies how to retrieve/assign value of this field.
                property(
                    # We need to bound the attribute name and field type here
                    # for the getter and setter of each field.
                    fget=partial(entry_getter, attr_name=name),
                    fset=partial(entry_setter, attr_name=name),
                ),
            )

        # Record that this entry hasn't been added to the index yet.
        self._pending_entries[entry.tid] = c

    # TODO: how to make this return the precise type here?
    def get_entry(self, tid: int) -> EntryType:
        r"""Look up the entry_index with ``tid``. Specific implementation
        depends on the actual class."""
        try:
            # Try to find entry in DataIndex
            entry: EntryType = self._index.get_entry(tid)
        except KeyError:
            # Find entry in DataStore
            entry = self._get_entry_from_data_store(tid=tid)
        if entry is None:
            raise KeyError(
                f"There is no entry with tid '{tid}'' in this datapack"
            )
        return entry

    def get_entry_raw(self, tid: int) -> List:
        r"""Retrieve the raw entry data in list format from DataStore."""
        return self._data_store.get_entry(tid=tid)[0]

    @abstractmethod
    def _save_entry_to_data_store(self, entry: Entry):
        r"""Save an existing entry object into DataStore"""
        raise NotImplementedError

    @abstractmethod
    def _get_entry_from_data_store(self, tid: int) -> EntryType:
        r"""Generate a class object from entry data in DataStore"""
        raise NotImplementedError

    @property
    @abstractmethod
    def links(self):
        r"""A List container of all links in this data pack."""
        raise NotImplementedError

    @property
    @abstractmethod
    def groups(self):
        r"""A List container of all groups in this pack."""
        raise NotImplementedError

    @abstractmethod
    def get_data(
        self, context_type, request, skip_k
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get(
        self, entry_type: Union[str, Type[EntryType]], **kwargs
    ) -> Iterator[EntryType]:
        """
        Implementation of this method should provide to obtain the entries in
        entry ordering. If there are orders defined between the entries, they
        should be used first. Otherwise, the insertion order should be
        used (FIFO).

        Args:
            entry_type: The type of the entry to obtain.

        Returns:
            An iterator of the entries matching the provided arguments.
        """
        raise NotImplementedError

    def get_single(self, entry_type: Union[str, Type[EntryType]]) -> EntryType:
        r"""Take a single entry of type
        :attr:`~forte.data.data_pack.DataPack.entry_type` from this data
        pack. This is useful when the target entry type appears only one
        time in the :class:`~forte.data.data_pack.DataPack` for e.g., a Document entry. Or you just
        intended to take the first one.

        Args:
            entry_type: The entry type to be retrieved.

        Returns:
            A single data entry.
        """
        for a in self.get(entry_type):
            return a

        raise EntryNotFoundError(
            f"The entry {entry_type} is not found in the provided pack."
        )

    def get_ids_by_creator(self, component: str) -> Set[int]:
        r"""
        Look up the component_index with key `component`. This will return
        the entry ids that are created by the `component`

        Args:
            component: The component (creator) to find ids for.

        Returns:
            A set of entry ids that are created by the component.
        """
        entry_set: Set[int] = self._creation_records[component]
        return entry_set

    def is_created_by(
        self, entry: Union[Entry, int], components: Union[str, Iterable[str]]
    ) -> bool:
        """
        Check if the entry is created by any of the provided components.

        Args:
            entry: `tid` of the entry or the entry object to check
            components: The list of component names.

        Returns:
            True if the entry is created by the component, False otherwise.
        """
        if isinstance(components, str):
            components = [components]

        entry_tid = entry.tid if isinstance(entry, Entry) else entry

        for c in components:
            if entry_tid in self._creation_records[c]:
                break
        else:
            # The entry not created by any of these components.
            return False

        return True

    def get_entries_from(self, component: str) -> Set[EntryType]:
        """
        Look up all entries from the `component` as a unordered set

        Args:
            component: The component (creator) to get the entries. It is
                normally the full qualified name of the creator class, but it
                may also be customized based on the implementation.

        Returns:
            The set of entry ids that are created by the input component.
        """
        return {
            self.get_entry(tid) for tid in self.get_ids_by_creator(component)
        }

    def get_ids_from(self, components: List[str]) -> Set[int]:
        """
        Look up entries using a list of components (creators). This will find
        each creator iteratively and combine the result.

        Args:
            components: The list of components to find.

        Returns:
            The list of entry ids that are created from these components.
        """
        valid_component_id: Set[int] = set()
        for component in components:
            valid_component_id |= self.get_ids_by_creator(component)
        return valid_component_id

    def _expand_to_sub_types(self, entry_type: Type[EntryType]) -> Set[Type]:
        """
        Return all the types and the sub types that inherit from the provided
        type.

        Args:
            entry_type: The provided type to search for entry.

        Returns:
            A set of all the sub-types extending the provided type, including
            the input ``entry_type`` itself.
        """
        all_types: Set[Type] = set()
        for data_type in self._index.indexed_types():
            if issubclass(data_type, entry_type):
                all_types.add(data_type)
        return all_types

    def get_entries_of(
        self, entry_type: Type[EntryType], exclude_sub_types=False
    ) -> Iterator[EntryType]:
        """
        Return all entries of this particular type without orders. If you
        need to get the annotations based on the entry ordering,
        use :meth:`forte.data.base_pack.BasePack.get`.

        Args:
            entry_type: The type of the entry you are looking for.
            exclude_sub_types: Whether to ignore the inherited sub type
                of the provided `entry_type`. Default is True.

        Returns:
            An iterator of the entries matching the type constraint.
        """
        if exclude_sub_types:
            for tid in self._index.query_by_type(entry_type):
                yield self.get_entry(tid)
        else:
            for tid in self._index.query_by_type_subtype(entry_type):
                yield self.get_entry(tid)

    @classmethod
    @abstractmethod
    def validate_link(cls, entry: EntryType) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def validate_group(cls, entry: EntryType) -> bool:
        raise NotImplementedError

    def get_links_from_node(
        self, node: Union[int, EntryType], as_parent: bool
    ) -> List[LinkType]:
        links: List[LinkType] = []
        if isinstance(node, Entry):
            tid = node.tid
            if tid is None:
                raise ValueError(
                    "The requested node has no tid. "
                    "Have you add this entry into the datapack?"
                )
        elif isinstance(node, int):
            tid = node
        else:
            raise TypeError(
                "Can only get group via entry id (int) or the "
                "group object itself (Entry)."
            )

        if not self._index.link_index_on:
            self._index.build_link_index(self.links)

        for tid in self._index.link_index(tid, as_parent=as_parent):
            entry: EntryType = self.get_entry(tid)
            if self.validate_link(entry):
                links.append(entry)  # type: ignore
        return links

    def get_links_by_parent(
        self, parent: Union[int, EntryType]
    ) -> List[LinkType]:
        return self.get_links_from_node(parent, True)

    def get_links_by_child(
        self, child: Union[int, EntryType]
    ) -> List[LinkType]:
        return self.get_links_from_node(child, False)

    def get_groups_by_member(
        self, member: Union[int, EntryType]
    ) -> Set[GroupType]:
        groups: Set[GroupType] = set()
        if isinstance(member, Entry):
            tid = member.tid
            if tid is None:
                raise ValueError(
                    "Argument member has no tid. "
                    "Have you add this entry into the datapack?"
                )
        elif isinstance(member, int):
            tid = member
        else:
            raise TypeError(
                "Can only get group via entry id (int) or the "
                "group object itself (Entry)."
            )

        if not self._index.group_index_on:
            self._index.build_group_index(self.groups)

        for tid in self._index.group_index(tid):
            entry: EntryType = self.get_entry(tid)
            if self.validate_group(entry):
                groups.add(entry)  # type: ignore
        return groups


PackType = TypeVar("PackType", bound=BasePack)
