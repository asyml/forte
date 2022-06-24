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
from typing import List, Iterator, Tuple, Any, Optional, Dict, Type
import json
from forte.data.ontology.core import Entry

__all__ = ["BaseStore"]


class BaseStore:
    # pylint: disable=too-many-public-methods
    r"""The base class which will be used by :class:`~forte.data.data_store.DataStore`."""

    def __init__(self):
        r"""
        This is a base class for the efficient underlying data structure. A
        current implementation of ``BaseStore`` is ``DataStore``.

        A ``BaseStore`` contains a collection of Forte entries.
        Each entry type contains some subtypes, which could have
        various fields stored in entry lists.
        """

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def serialize(
        self,
        output_path: str,
        serialize_method: str = "json",
        save_attribute: bool = True,
        indent: Optional[int] = None,
    ):
        """
        Serializes the data store to the provided path. The output of this
        function depends on the serialization method chosen.

        Args:
            output_path: The path to write data to.
            serialize_method: The method used to serialize the data. Currently
                supports `json` (outputs json dictionary).
            save_attribute: Boolean value indicating whether users want to
                save attributes for field checks later during deserialization.
                Attributes and their indices for every entry type will be saved.
            indent: Whether to indent the file if written as JSON.

        Returns: Results of serialization.
        """
        if serialize_method == "json":
            with open(output_path, mode="wt", encoding="utf-8") as json_out:
                json_out.write(
                    self.to_string(serialize_method, save_attribute, indent)
                )
        else:
            raise NotImplementedError(
                f"Unsupported serialization method {serialize_method}"
            )

    def to_string(
        self,
        json_method: str = "json",
        save_attribute: bool = True,
        indent: Optional[int] = None,
    ) -> str:
        """
        Return the string representation (json encoded) of this method.

        Args:
            json_method: What method is used to convert data pack to json.
                Only supports `json` for now. Default value is `json`.
            save_attribute: Boolean value indicating whether users want to
                save attributes for field checks later during deserialization.
                Attributes and their indices for every entry type will be saved.
        Returns: String representation of the data pack.
        """
        if json_method == "json":
            state = self.__getstate__()
            if not save_attribute:
                state.pop("fields")
            return json.dumps(state, indent=indent)
        else:
            raise ValueError(f"Unsupported JSON method {json_method}.")

    @classmethod
    def _deserialize(
        cls,
        data_source: str,
        serialize_method: str = "json",
    ) -> Dict:
        """
        This function should deserialize a data store from a string.

        Args:
            data_source: The data path containing data store. The content
                of the data could be string or bytes depending on the method of
                serialization.
            serialize_method: The method used to serialize the data, this
                should be the same as how serialization is done. The current
                option is `json`。

        Returns:
            The state of the data store object deserialized from the data.
        """
        if serialize_method == "json":
            with open(data_source, mode="rt", encoding="utf8") as f:
                state = json.loads(f.read())
            return state
        else:
            raise NotImplementedError(
                f"Unsupported deserialization method {serialize_method}"
            )

    @abstractmethod
    def add_entry_raw(
        self,
        type_name: str,
        attribute_data: List,
        base_class: Type[Entry],
        tid: Optional[int] = None,
        allow_duplicate: bool = True,
    ) -> int:

        r"""
        This function provides a general implementation to add all
        types of entries to the data store. It can add namely
        Annotation, AudioAnnotation, ImageAnnotation,
        Link, Group and Generics. Returns the ``tid`` for the
        inserted entry.

        Args:
            type_name: The fully qualified type name of the new Entry.
            attribute_data: It is a list that stores attributes relevant to
                the entry being added. In order to keep the number of attributes
                same for all entries, the list is populated with trailing None's.
            base_class: The type of entry to add to the Data Store. This is
                a reference to the class of the entry that needs to be added
                to the DataStore. The reference can be to any of the classes
                supported by the function.
            tid: ``tid`` of the Entry that is being added.
                It's optional, and it will be
                auto-assigned if not given.
            allow_duplicate: Whether we allow duplicate in the DataStore. When
                it's set to False, the function will return the ``tid`` of
                existing entry if a duplicate is found. Default value is True.

        Returns:
            ``tid`` of the entry.
        """
        raise NotImplementedError

    @abstractmethod
    def add_annotation_raw(
        self,
        type_name: str,
        begin: int,
        end: int,
        tid: Optional[int] = None,
        allow_duplicate: bool = True,
    ) -> int:
        r"""This function adds an annotation entry with ``begin`` and ``end``
        indices to the ``type_name`` sorted list in ``self.__elements``,
        returns the ``tid`` for the inserted entry.

        Args:
            type_name: The index of Annotation sorted list in ``self.__elements``.
            begin: Begin index of the entry.
            end: End index of the entry.
            tid: ``tid`` of the Annotation entry that is being added.
                It's optional, and it will be auto-assigned if not given.
            allow_duplicate: Whether we allow duplicate in the DataStore. When
                it's set to False, the function will return the ``tid`` of
                existing entry if a duplicate is found. Default value is True.
        Returns:
            ``tid`` of the entry.
        """
        raise NotImplementedError

    @abstractmethod
    def add_link_raw(
        self,
        type_name: str,
        parent_tid: int,
        child_tid: int,
        tid: Optional[int] = None,
    ) -> Tuple[int, int]:
        r"""This function adds a link entry with ``parent_tid`` and ``child_tid``
        to the ``type_name`` list in ``self.__elements``, returns the ``tid`` and the
        ``index_id`` for the inserted entry in the list. This ``index_id`` is the
        index of the entry in the ``type_name`` list.

        Args:
            type_name: The index of Link list in ``self.__elements``.
            parent_tid: ``tid`` of the parent entry.
            child_tid: ``tid`` of the child entry.
            tid: ``tid`` of the Link entry that is being added.
                It's optional, and it will be auto-assigned if not given.

        Returns:
            ``tid`` of the entry and its index in the ``type_name`` list.

        """
        raise NotImplementedError

    @abstractmethod
    def add_group_raw(
        self, type_name: str, member_type: str, tid: Optional[int] = None
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
    def add_generics_raw(
        self, type_name: str, tid: Optional[int] = None
    ) -> Tuple[int, int]:
        r"""This function adds a generics entry with ``type_name`` to the
        current data store object. Returns the ``tid`` and the ``index_id``
        for the inserted entry in the list. This ``index_id`` is the index
        of the entry in the ``type_name`` list.

        Args:
            type_name: The fully qualified type name of the new Generics.
            tid: ``tid`` of generics entry.

        Returns:
            ``tid`` of the entry and its index in the (``type_id``)th list.

        """
        raise NotImplementedError

    @abstractmethod
    def add_audio_annotation_raw(
        self,
        type_name: str,
        begin: int,
        end: int,
        tid: Optional[int] = None,
        allow_duplicate=True,
    ) -> int:

        r"""
        This function adds an audio annotation entry with ``begin`` and ``end``
        indices to current data store object. Returns the ``tid`` for the
        inserted entry.

        Args:
            type_name: The fully qualified type name of the new AudioAnnotation.
            begin: Begin index of the entry.
            end: End index of the entry.
            tid: ``tid`` of the Annotation entry that is being added.
                It's optional, and it will be
                auto-assigned if not given.
            allow_duplicate: Whether we allow duplicate in the DataStore. When
                it's set to False, the function will return the ``tid`` of
                existing entry if a duplicate is found. Default value is True.

        Returns:
            ``tid`` of the entry.
        """
        raise NotImplementedError

    @abstractmethod
    def add_image_annotation_raw(
        self,
        type_name: str,
        image_payload_idx: int,
        tid: Optional[int] = None,
    ) -> int:

        r"""
        This function adds an image annotation entry with ``image_payload_idx``
        indices to current data store object. Returns the ``tid`` for the
        inserted entry.

        Args:
            type_name: The fully qualified type name of the new AudioAnnotation.
            image_payload_idx: the index of the image payload.
            tid: ``tid`` of the Annotation entry that is being added.
                It's optional, and it will be
                auto-assigned if not given.

        Returns:
            ``tid`` of the entry.
        """
        raise NotImplementedError

    @abstractmethod
    def add_multipack_generic_raw(
        self, type_name: str, tid: Optional[int] = None
    ) -> Tuple[int, int]:
        r"""This function adds a multi pack generic entry with ``type_name`` to
        the current data store object. Returns the ``tid`` and the ``index_id``
        for the inserted entry in the list. This ``index_id`` is the index
        of the entry in the ``type_name`` list.

        Args:
            type_name: The fully qualified type name of the new Generics.
            tid: ``tid`` of multi pack generic entry.

        Returns:
            ``tid`` of the entry and its index in the (``type_id``)th list.

        """
        raise NotImplementedError

    @abstractmethod
    def add_multipack_link_raw(
        self,
        type_name: str,
        parent_pack_id: int,
        parent_tid: int,
        child_pack_id: int,
        child_tid: int,
        tid: Optional[int] = None,
    ) -> Tuple[int, int]:
        r"""This function adds a multi pack link entry with ``parent_tid`` and
        ``child_tid`` to current data store object. Returns the ``tid`` and
        the ``index_id`` for the inserted entry in the list. This ``index_id``
        is the index of the entry in the ``type_name`` list.

        Args:
            type_name:  The fully qualified type name of the new
                ``MultiPackLink``.
            parent_pack_id: ``pack_id`` of the parent entry.
            parent_tid: ``tid`` of the parent entry.
            child_pack_id: ``pack_id`` of the child entry.
            child_tid: ``tid`` of the child entry.
            tid: ``tid`` of the ``MultiPackLink`` entry that is being added.
                It's optional, and it will be auto-assigned if not given.

        Returns:
            ``tid`` of the entry and its index in the ``type_name`` list.
        """
        raise NotImplementedError

    @abstractmethod
    def add_multipack_group_raw(
        self, type_name: str, member_type: str, tid: Optional[int] = None
    ) -> Tuple[int, int]:
        r"""This function adds a multi pack group entry with ``member_type`` to
        the current data store object. Returns the ``tid`` and the ``index_id``
        for the inserted entry in the list. This ``index_id`` is the index
        of the entry in the ``type_name`` list.

        Args:
            type_name: The fully qualified type name of the new
                ``MultiPackGroup``.
            member_type: Fully qualified name of its members.
            tid: ``tid`` of the ``MultiPackGroup`` entry that is being added.
                It's optional, and it will be auto-assigned if not given.

        Returns:
            ``tid`` of the entry and its index in the (``type_id``)th list.
        """
        raise NotImplementedError

    @abstractmethod
    def all_entries(self, entry_type_name: str) -> Iterator[List]:
        """
        Retrieve all entry data of entry type ``entry_type_name`` and
        entries of subclasses of entry type ``entry_type_name``.

        Args:
            entry_type_name (str): the type name of entries that the User wants to retrieve.

        Yields:
            Iterator of raw entry data in list format.
        """
        raise NotImplementedError

    @abstractmethod
    def num_entries(self, entry_type_name: str) -> int:
        """
        Compute the number of entries of given ``entry_type_name`` and
        entries of subclasses of entry type ``entry_type_name``.

        Args:
            entry_type_name (str): the type name of entries that the User wants to get its count.

        Returns:
            The number of entries of given ``entry_type_name``.
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
    def _set_attr(self, tid: int, attr_id: int, attr_value: Any):
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
    def _get_attr(self, tid: int, attr_id: int):
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

    @abstractmethod
    def _is_subclass(
        self, type_name: str, cls, no_dynamic_subclass: bool = False
    ) -> bool:
        r"""This function takes a fully qualified ``type_name`` class name,
        ``cls`` class and returns whether ``type_name``  class is the``cls``
        subclass or not. This function accepts two types of class: the class defined
        in forte, or the classes in user provided ontology file.


        Args:
            type_name: A fully qualified name of an entry class.
            cls: An entry class.
            no_dynamic_subclass: A boolean value controlling where to look for
            subclasses. If True, this function will not check the subclass
            relations via `issubclass` but rely on pre-populated states only.

        Returns:
            A boolean value whether ``type_name``  class is the``cls``
            subclass or not.

        """
        raise NotImplementedError

    @abstractmethod
    def _is_annotation(self, type_name: str) -> bool:
        r"""This function takes a type_name and returns whether a type
        is an annotation type or not.
        Args:
            type_name: The name of type in `self.__elements`.

        Returns:
            A boolean value whether this type_name belongs to an annotation
            type or not.
        """
        raise NotImplementedError
