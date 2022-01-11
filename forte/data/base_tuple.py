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

__all__ = ["BaseTuple"]


class BaseTuple(EntryContainer[EntryType, LinkType, GroupType]):
    r"""The base class of :class:`~forte.data.data_pack.DataTuple`.

    Args:
        tuple_name (str, optional): a string name of the data tuple.

    """

    # pylint: disable=too-many-public-methods
    def __init__(self, tuple_name: Optional[str] = None):
        super().__init__()

        # A dictionary that records all entrys with structure {tid: entry}.
        self.entry_dict: dict = dict()

        # A sorted list of (class_name, begin, end, tid, speaker, part_id,
        #                   sentiment, classification, classifications)
        # If the field has not been initialized, we use None as a place holder.
        # Need to implement all types of entries as separate lists
        self.sentence: List[tuple] = []

        # Field dictionaries which record all possible fields
        # Need to implement all types of entries as separate dictionaries
        self.sentence_field_dict = {
            "speaker": 4,
            "part_id": 5,
            "sentiment": 6,
            "classification": 7,
            "classifications": 8,
        }

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
        Serializes the Tuple to the provided path. The output of this
        function depends on the serialization method chosen.

        We want to further optimize this function.

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
    ):
        """
        This function should deserialize a Tuple from a string. The
        implementation should decide the specific pack type.

        We want to further optimize this function.

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

    @classmethod
    def from_string(cls, data_content: str) -> "BaseTuple":
        return jsonpickle.decode(data_content)

    @abstractmethod
    def add_entry_raw(self, entry_type, begin, end):
        r"""Add an entry to the data_tuple and return a unique tid for it.
        We need to add the entry to the corresponding sorted list and the
        entry_dict. We assign a tid to it.

        Args:
            entry_type (EntryType): The type of the entry to be added.
            begin (int): begin index of the entry.
            end (int): end index of the entry.

        Returns:
            The tid of the entry.

        """

        raise NotImplementedError

    @abstractmethod
    def set_attr(self, tid, attr_name, attr_value):
        r"""Set the attribute `attr_name` of an entry with value `attr_value`.
        We first retrieve the entry and find its type. We then locate the index
        of the attribute from the field dictionary. We need to update the tuple
        in both entry_dict and the corresponding sorted list.

        Args:
            tid (int): Unique id of the entry.
            attr_name (str): name of the attribute.
            attr_value: value of the attribute.

        Returns:

        """
        raise NotImplementedError

    def get_attr(self, tid, attr_name) -> List:
        r"""Get the value of `attr_name` of an entry.

        Args:
            tid (int): Unique id of the entry.
            attr_name (str): name of the attribute.

        Returns:

        """
        entry = self.entry_dict[tid]
        return self.get_attr_from_tuple(entry, attr_name)

    @abstractmethod
    def get_attr_from_tuple(self, entry, attr_name):
        r"""Get the value of `attr_name` of an entry.
        We first find the type of the entry. We then locate the index of the
        attribute from the field dictionary and get the field.

        Args:
            entry (tuple): the entry we query.
            attr_name (str): name of the attribute.

        Returns:
            Value of the attribute.

        """
        raise NotImplementedError

    @abstractmethod
    def delete_entry(self, tid):
        r"""Remove the entry from the tuple.
        We locate the entry from entry_dict using tid and remove it from both
        entry_dict and the corresponding sorted list.

        Args:
            tid (int): Unique id of the entry.

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def get_entry(self, tid: int):
        r"""Look up the entry_dict with key `tid`.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The entry which tid corresponds to.

        """
        raise NotImplementedError

    @abstractmethod
    def get_data_raw(
        self, context_type, request, skip_k
    ) -> Iterator[Dict[str, Any]]:
        r"""Fetch entries from the data_tuple of type `context_type`.

        Example:

            .. code-block:: python

                requests = {
                    base_ontology.Sentence:
                        {
                            "component": ["dummy"],
                            "fields": ["speaker"],
                        },
                    base_ontology.Token: ["pos", "sense""],
                    base_ontology.EntityMention: {
                        "unit": "Token",
                    },
                }
                pack.get_data_raw(base_ontology.Sentence, requests)

        Args:
            context_type (str or EntryType): The granularity of the data
                context, which could be any ``Annotation`` type.
            request (dict): The entry types and fields required.
                The keys of the requests dict are the required entry types
                and the value should be either:

                - a list of field names or
                - a dict which accepts three keys: `"fields"`, `"component"`,
                  and `"unit"`.

                    - By setting `"fields"` (list), users
                      specify the requested fields of the entry. If "fields"
                      is not specified, only the default fields will be
                      returned.
                    - By setting `"component"` (list), users
                      can specify the components by which the entries are
                      generated. If `"component"` is not specified, will return
                      entries generated by all components.
                    - By setting `"unit"` (string), users can
                      specify a unit by which the annotations are indexed.

                Note that for all annotation types, `"text"` and `"span"`
                fields are returned by default; for all link types, `"child"`
                and `"parent"` fields are returned by default.
            skip_k (int): Will skip the first `skip_k` instances and generate
                data from the (`offset` + 1)th instance.

        Returns:
            A data generator, which generates one piece of data (a dict
            containing the required entries, fields, and context).

        """
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

    @abstractmethod
    def next_entry(self, tid):
        r"""Get the next entry in the order of the sorted list.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The next entry of the same type as the tid entry.

        """
        raise NotImplementedError

    @abstractmethod
    def prev_entry(self, tid):
        r"""Get the previous entry in the order of the sorted list.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The previous entry of the same type as the tid entry.

        """
        raise NotImplementedError

    def get_single(self, entry_type: Union[str, Type[EntryType]]) -> EntryType:
        r"""Take a single entry of type :attr:`entry_type` from this data
        tuple. This is useful when the target entry type appears only one
        time in the :class:`DataTuple` for e.g., a Document entry. Or you just
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

    """
    old functions from data pack class
    """

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
