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
from typing import List, Type, Union, Iterator, Tuple, Any

from forte.data.ontology.core import EntryType

__all__ = ["BaseStore"]


class BaseStore:
    r"""The base class which will be used by :class:
    `~forte.data.data_store.DataStore`."""

    def __init__(self):
        r"""
        This is a base class for the efficient underlying data structure. A
        current implementation of `BaseStore` is `DataStore`.

        A `BaseStore` contains a collection of Forte entries.
        Each entry type contains some subtypes, which could have
        various fields stored in entry lists.
        """

    @abstractmethod
    def add_annotation_raw(
        self, entry_type: str, begin: int, end: int
    ) -> Tuple[int, int]:
        r"""This function adds an annotation entry with `begin` and `end`
        indices to the `entry_type` sortedlist, returns the `tid` and the
        2-D index for the inserted entry.

        Args:
            entry_type (str): Fully qualified name of this annotation.
            begin (int): Begin index of the entry.
            end (int): End index of the entry.

        Returns:
            `tid` of the entry and its index in the entry list.

        """
        raise NotImplementedError

    @abstractmethod
    def add_link_raw(
        self, entry_type: str, parent_tid: int, child_tid: int
    ) -> Tuple[int, int]:
        r"""This function adds a link entry with `parent_tid` and `child_tid`
        to the `entry_type` list, returns the `tid` and the 2-D index for the
        inserted entry.

        Args:
            entry_type (str): Fully qualified name of this link.
            parent_tid (int): `tid` of the parent entry.
            child_tid (int): `tid` of the child entry.

        Returns:
            `tid` of the entry and its index in the entry list.

        """
        raise NotImplementedError

    @abstractmethod
    def add_group_raw(
        self, entry_type: str, member_type: str
    ) -> Tuple[int, int]:
        r"""This function adds a group entry with `member_type` to the
        `entry_type` list, returns the `tid` and the 2-D index for the
        inserted entry.

        Args:
            entry_type (str): Fully qualified name of this group.
            member_type (str): Fully qualified name of its members.

        Returns:
            `tid` of the entry and its index in the entry list.

        """
        raise NotImplementedError

    @abstractmethod
    def set_attribute(self, tid: int, attr_name: str, attr_value: Any):
        r"""This function locates the entry data with `tid` and sets its
        `attr_name` with `attr_value`.

        Args:
            tid (int): Unique Id of the entry.
            attr_name (str): Name of the attribute.
            attr_value (any): Value of the attribute.
        """
        raise NotImplementedError

    @abstractmethod
    def set_attr(self, tid: int, attr_id: int, attr_value: Any):
        r"""This function locates the entry data with `tid` and sets its
        attribute `attr_id` with value `attr_value`.
        Called by `set_attribute()`.

        Args:
            tid (int): Unique id of the entry.
            attr_id (int): Id of the attribute.
            attr_value: value of the attribute.

        """

        raise NotImplementedError

    @abstractmethod
    def get_attribute(self, tid: int, attr_name: str):
        r"""This function finds the value of `attr_name` in entry with
        `tid`.

        Args:
            tid (int): Unique id of the entry.
            attr_name (str): Name of the attribute.

        Returns:
            The value of `attr_name` for the entry with `tid`.
        """
        raise NotImplementedError

    @abstractmethod
    def get_attr(self, tid: int, attr_id: int):
        r"""This function locates the entry data with `tid` and gets the value
        of `attr_id` of this entry. Called by `get_attribute()`.

        Args:
            tid (int): Unique id of the entry.
            attr_id (int): Id of the attribute.

        Returns:
            The value of `attr_id` for the entry with `tid`.
        """

        raise NotImplementedError

    @abstractmethod
    def delete_entry(self, tid: int):
        r"""This function removes the entry with `tid` from the data store.

        Args:
            tid (int): Unique id of the entry.

        """

        raise NotImplementedError

    @abstractmethod
    def get_entry(self, tid: int) -> Tuple[List, int, int]:
        r"""Look up the entry_dict with key `tid`. Find its type_id and its
        index in the `entry_type` list.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The entry which `tid` corresponds to.

        """
        raise NotImplementedError

    @abstractmethod
    def get(
        self, entry_type: Union[str, Type[EntryType]], **kwargs
    ) -> Iterator[List]:
        """
        Implementation of this method should provide to obtain the entries of
        type `entry_type`.

        Args:
            entry_type: Fully qualified name of this annotation.

        Returns:
            An iterator of the entries matching the provided arguments.

        """

        raise NotImplementedError

    @abstractmethod
    def next_entry(self, tid: int) -> List:
        r"""Get the next entry of the same type as the `tid` entry.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The next entry of the same type as the `tid` entry.

        """

        raise NotImplementedError

    @abstractmethod
    def prev_entry(self, tid: int) -> List:
        r"""Get the previous entry of the same type as the `tid` entry.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The previous entry of the same type as the `tid` entry.

        """

        raise NotImplementedError
