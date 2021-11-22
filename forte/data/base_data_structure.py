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

import copy
import gzip
import pickle
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import (
    AbstractSet,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    Iterator,
    Dict,
    Tuple,
    Any,
    Iterable,
)

import jsonpickle

from forte.common import ProcessExecutionException, EntryNotFoundError
from forte.data.container import EntryContainer
from forte.data.index import BaseIndex
from forte.data.ontology.core import Entry, EntryType, GroupType, LinkType

__all__ = ["BaseDataStructure"]

class BaseDataStructure(EntryContainer[EntryType, LinkType, GroupType]):
    r"""The base class of :class:`~forte.data.data_pack.DataTuple`.

    Args:
        pack_name (str, optional): a string name of the pack.

    """

    # pylint: disable=too-many-public-methods
    def __init__(self, pack_name: Optional[str] = None):
        super().__init__()

    @abstractmethod
    def __iter__(self) -> Iterator[EntryType]:
        raise NotImplementedError
    
    def to_string(
        self,
        drop_record: Optional[bool] = False,
        json_method: str = "jsonpickle",
        indent: Optional[int] = None,
    ) -> str:
        """
        Return the string representation (json encoded) of this method.

        Args:
            drop_record: Whether to drop the creation records, default is False.
            json_method: What method is used to convert data pack to json.
              Only supports `json_pickle` for now. Default value is
              `json_pickle`.
            indent: The indent used for json string.

        Returns: String representation of the data pack.
        """
        if drop_record:
            self._creation_records.clear()
            self._field_records.clear()
        if json_method == "jsonpickle":
            return jsonpickle.encode(self, unpicklable=True, indent=indent)
        else:
            raise ValueError(f"Unsupported JSON method {json_method}.")

    def serialize(
        self,
        output_path: Union[str, Path],
        zip_pack: bool = False,
        drop_record: bool = False,
        serialize_method: str = "jsonpickle",
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
              supports "jsonpickle" (outputs str) and Python's built-in
              "pickle" (outputs bytes).
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
                pickle.dump(self, pickle_out)  # type:ignore
        elif serialize_method == "jsonpickle":
            with _open(output_path, mode="wt", encoding="utf-8") as json_out:
                json_out.write(
                    self.to_string(drop_record, "jsonpickle", indent=indent)
                )
        else:
            raise NotImplementedError(
                f"Unsupported serialization method {serialize_method}"
            )

    @classmethod
    def deserialize(
        cls,
        data_source: Union[Path, str],
        serialize_method: str = "jsonpickle",
        zip_pack: bool = False,
    ) -> "PackType":
        """
        This function should deserialize a Pack from a string. The
        implementation should decide the specific pack type.

        Args:
            data_source: The data path containing pack data. The content
              of the data could be string or bytes depending on the method of
              serialization.
            serialize_method: The method used to serialize the data, this
              should be the same as how serialization is done. The current
              options are "jsonpickle" and "pickle". The default method
              is "jsonpickle".
            zip_pack: Boolean value indicating whether the input source is
              zipped.

        Returns:
            An pack object deserialized from the data.
        """
        _open = gzip.open if zip_pack else open

        if serialize_method == "jsonpickle":
            with _open(data_source, mode="rt") as f:  # type: ignore
                pack = cls.from_string(f.read())
        else:
            with _open(data_source, mode="rb") as f:  # type: ignore
                pack = pickle.load(f)

        return pack  # type: ignore
    
    def on_entry_creation(self, entry):
        pass

    def regret_creation(self, entry):
        pass

    def record_field(self, entry_id: int, field_name: str):
        pass

    def add_all_remaining_entries(self):
        pass
    
    def set_control_component(self, component):
        pass

    @classmethod
    def from_string(cls, data_content: str) -> "BaseDataStructure":
        return jsonpickle.decode(data_content)

    @abstractmethod
    def add_entry_raw(self, entry_type, begin, end):
        raise NotImplementedError
    
    @abstractmethod
    def set_attr(self, tid, attr_name, attr_value):
        raise NotImplementedError
    
    @abstractmethod
    def get_attr(self, tid, attr_name):
        raise NotImplementedError

    @abstractmethod
    def get_attr_from_tuple(self, entry, attr_name):
        raise NotImplementedError

    @abstractmethod
    def delete_entry(self, entry: EntryType):
        r"""Remove the entry from the pack.

        Args:
            entry: The entry to be removed.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def get_entry(self, tid: int):
        r"""Look up the entry_index with key ``ptr``. Specific implementation
        depends on the actual class."""
        raise NotImplementedError

    @abstractmethod
    def get_data_raw(
        self, context_type, request, skip_k
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_raw(
        self, entry_type: Union[str, Type[EntryType]], **kwargs
    ) -> Iterator[Tuple]:
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
        r"""Take a single entry of type :attr:`entry_type` from this data
        pack. This is useful when the target entry type appears only one
        time in the :class:`DataPack` for e.g., a Document entry. Or you just
        intended to take the first one.

        Args:
            entry_type: The entry type to be retrieved.

        Returns:
            A single data entry.
        """
        for a in self.get_raw(entry_type):
            return a

        raise EntryNotFoundError(
            f"The entry {entry_type} is not found in the provided pack."
        )

    def _expand_to_sub_types(self, entry_type: Type[EntryType]) -> Set[Type]:
        """
        Return all the types and the sub types that inherit from the provided
        type.

        Args:
            entry_type: The provided type to search for entry.

        Returns:
            A set of all the sub-types extending the provided type, including
            the input `entry_type` itself.
        """
        all_types: Set[Type] = set()
        # for data_type in self._index.indexed_types():
        #     if issubclass(data_type, entry_type):
        #         all_types.add(data_type)
        all_types.add(entry_type)
        return all_types

    def get_entries_of(
        self, entry_type: Type[EntryType], exclude_sub_types=False
    ) -> Iterator[EntryType]:
        """
        Return all entries of this particular type without orders. If you
        need to get the annotations based on the entry ordering,
        use :meth:`forte.data.base_pack.get`.

        Args:
            entry_type: The type of the entry you are looking for.
            exclude_sub_types (bool): Whether to ignore the inherited sub type
            of the provided `entry_type`. Default is True.

        Returns:
            An iterator of the entries matching the type constraint.
        """
        return None