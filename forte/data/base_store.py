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
from typing import List, Iterator, Tuple, Any, Optional

__all__ = ["BaseStore"]


class BaseStore:
    r"""The base class which will be used by :class:`~forte.data.data_store.DataStore`."""

    def __init__(self):
        r"""
        This is a base class for the efficient underlying data structure. A
        current implementation of ``BaseStore`` is ``DataStore``.

        A ``BaseStore`` contains a collection of Forte entries.
        Each entry type contains some subtypes, which could have
        various fields stored in entry lists.
        """

    @abstractmethod
    def add_annotation_raw(self, type_name: str, begin: int, end: int) -> int:
        r"""This function adds an annotation entry with ``begin`` and ``end``
        indices to the ``type_name`` sorted list in ``self.__elements``,
        returns the ``tid`` for the inserted entry.

        Args:
            type_name: The index of Annotation sorted list in ``self.__elements``.
            begin: Begin index of the entry.
            end: End index of the entry.
        Returns:
            ``tid`` of the entry.
        """
        raise NotImplementedError

    @abstractmethod
    def add_link_raw(
        self, type_name: str, parent_tid: int, child_tid: int
    ) -> Tuple[int, int]:
        r"""This function adds a link entry with ``parent_tid`` and ``child_tid``
        to the ``type_name`` list in ``self.__elements``, returns the ``tid`` and the
        ``index_id`` for the inserted entry in the list. This ``index_id`` is the
        index of the entry in the ``type_name`` list.

        Args:
            type_name: The index of Link list in ``self.__elements``.
            parent_tid: ``tid`` of the parent entry.
            child_tid: ``tid`` of the child entry.

        Returns:
            ``tid`` of the entry and its index in the ``type_name`` list.

        """
        raise NotImplementedError

    @abstractmethod
    def add_group_raw(
        self, type_name: str, member_type: str
    ) -> Tuple[int, int]:
        r"""This function adds a group entry with ``member_type`` to the
        ``type_name`` list in ``self.__elements``, returns the ``tid`` and the
        ``index_id`` for the inserted entry in the list. This ``index_id`` is the
        index of the entry in the ``type_name`` list.

        Args:
            type_name: The index of Group list in ``self.__elements``.
            member_type: Fully qualified name of its members.

        Returns:
            ``tid`` of the entry and its index in the ``type_name`` list.

        """
        raise NotImplementedError

    @abstractmethod
    def set_attribute(self, tid: int, attr_name: str, attr_value: Any):
        r"""This function locates the entry data with ``tid`` and sets its
        ``attr_name`` with ``attr_value``.

        Args:
            tid: Unique Id of the entry.
            attr_name: Name of the attribute.
            attr_value: Value of the attribute.
        """
        raise NotImplementedError

    @abstractmethod
    def set_attr(self, tid: int, attr_id: int, attr_value: Any):
        r"""This function locates the entry data with ``tid`` and sets its
        attribute ``attr_id``  with value ``attr_value``.
        Called by `set_attribute()`.

        Args:
            tid: Unique id of the entry.
            attr_id: Id of the attribute.
            attr_value: value of the attribute.

        """

        raise NotImplementedError

    @abstractmethod
    def get_attribute(self, tid: int, attr_name: str):
        r"""This function finds the value of ``attr_name`` in entry with
        ``tid``.

        Args:
            tid: Unique id of the entry.
            attr_name: Name of the attribute.

        Returns:
            The value of ``attr_name`` for the entry with ``tid``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_attr(self, tid: int, attr_id: int):
        r"""This function locates the entry data with ``tid`` and gets the value
        of ``attr_id``  of this entry. Called by `get_attribute()`.

        Args:
            tid: Unique id of the entry.
            attr_id: Id of the attribute.

        Returns:
            The value of ``attr_id``  for the entry with ``tid``.
        """

        raise NotImplementedError

    @abstractmethod
    def delete_entry(self, tid: int):
        r"""This function removes the entry with ``tid`` from the data store.

        Args:
            tid: Unique id of the entry.

        """

        raise NotImplementedError

    @abstractmethod
    def get_entry(self, tid: int) -> Tuple[List, str]:
        r"""Look up the `tid_ref_dict` or `tid_idx_dict` with key ``tid``.
        Return the entry and its ``type_name``.

        Args:
            tid: Unique id of the entry.

        Returns:
            The entry which ``tid`` corresponds to and its ``type_name``.

        """
        raise NotImplementedError

    @abstractmethod
    def get_entry_index(self, tid: int) -> int:
        r"""Look up the `tid_ref_dict` or `tid_idx_dict` with key ``tid``.
        Return the ``index_id`` of the entry.

        Args:
            tid: Unique id of the entry.

        Returns:
            Index of the entry which ``tid`` corresponds to in the
            ``entry_type`` list.

        """
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        type_name: str,
        include_sub_type: bool,
        range_annotation: Optional[Tuple[int]] = None,
    ) -> Iterator[List]:
        r"""This function fetches entries from the data store of
        type ``type_name``.

        Args:
            type_name: The index of the list in ``self.__elements``.
            include_sub_type: A boolean to indicate whether get its subclass.
            range_annotation: A tuple that contains the begin and end indices
                of the searching range of annotation-like entries.

        Returns:
            An iterator of the entries matching the provided arguments.

        """

        raise NotImplementedError

    @abstractmethod
    def next_entry(self, tid: int) -> Optional[List]:
        r"""Get the next entry of the same type as the ``tid`` entry.

        Args:
            tid: Unique id of the entry.

        Returns:
            The next entry of the same type as the ``tid`` entry.

        """

        raise NotImplementedError

    @abstractmethod
    def prev_entry(self, tid: int) -> Optional[List]:
        r"""Get the previous entry of the same type as the ``tid`` entry.

        Args:
            tid: Unique id of the entry.

        Returns:
            The previous entry of the same type as the ``tid`` entry.

        """

        raise NotImplementedError
