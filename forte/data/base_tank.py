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

from abc import abstractmethod
from typing import (
    List,
    Type,
    Union,
    Iterator,
    Dict,
    Tuple,
    Any,
)

from sortedcontainers import SortedList
from forte.common import EntryNotFoundError
from forte.data.ontology.core import EntryType

__all__ = ["BaseTank"]


class BaseTank:
    r"""The base class which will be
    used by :class:`~forte.data.data_pack.DataPack`."""

    # pylint: disable=too-many-public-methods
    def __init__(self):

        # A dictionary that records all entrys with structure {tid: entry}.
        self.entry_dict: dict = {}

        # A sorted list of (class_name, begin, end, tid, speaker, part_id,
        #                   sentiment, classification, classifications)
        # If the field has not been initialized, we use None as a place holder.
        # Need to implement all types of entries as separate lists
        self.sentence: SortedList[tuple] = SortedList(key=self.key_function)

        # Field dictionaries which record all possible fields
        # Need to implement all types of entries as separate dictionaries
        self.sentence_field = {
            "speaker": 4,
            "part_id": 5,
            "sentiment": 6,
            "classification": 7,
            "classifications": 8,
        }

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    def key_function(self, x):
        # return begin and end indexes of the tuple
        return x[1], x[2]

    @abstractmethod
    def add_entry_raw(self, entry_type, begin, end):
        r"""Add an entry to the data_tank and return a unique tid for it.
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
        r"""Remove the entry from the data tank.
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
        r"""Fetch entries from the data_tank of type `context_type`.

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
                tank.get_data_raw(base_ontology.Sentence, requests)

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

    def get_single(self, entry_type: Union[str, Type[EntryType]]) -> EntryType:
        r"""Take a single entry of type :attr:`entry_type` from this data
        tank. This is useful when the target entry type appears only one
        time in the :class:`DataTank` for e.g., a Document entry. Or you just
        intended to take the first one.

        Args:
            entry_type: The entry type to be retrieved.

        Returns:
            A single data entry.

        """
        for a in self.get_raw(entry_type):
            return a

        raise EntryNotFoundError(
            f"The entry {entry_type} is not found in the provided data tank."
        )

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
