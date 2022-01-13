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

from forte.common import EntryNotFoundError
from forte.data.ontology.core import EntryType

__all__ = ["BaseStore"]


class BaseStore:
    r"""The base class which will be used by :class:
    `~forte.data.data_pack.DataPack`."""

    # pylint: disable=too-many-public-methods
    def __init__(self):
        r"""This class stores entries by types in different SortedLists. It
        uses an array to store these SortedLists with the structure:
        array([Document SortedList, Sentence SortedList, ...]).
        Different types of annotations, like sentence, tokens and documents,
        are stored in different SortedLists.

        This class records the order of the array and assigns type id,
        which is the index of each sortedlist, to each entry type.

        Each entry list in the SortedList has the format of
        [entry_type, tid, begin, end, attr_1, attr_2, ..., attr_n]
        The first four fields are compulsory for every entry type.
        Each entry type has a fixed field of attributes.
        E.g. Document SortedList has lists of structure:
        [entry_type, tid, begin, end, document_class,sentiment,classifications]


        The type_attributes dictionay will be passed in. It provides entry
        types and their corresponding attribues. The keys are all valid
        ontology types as strings, including all the types defined in ft.onto
        and ftx.onto. The values are all the valid attributes for this type,
        also defined in ft.onto and ftx.onto.
        Example:
        type_attributes = {
            "Token": ["pos", "ud_xpos", "lemma", "chunk", "ner", "sense",
                    "is_root", "ud_features", "ud_misc"],
            "Document": ["document_class", "sentiment", "classifications"],
            "Sentence": ["speaker", "part_id", "sentiment", "classification",
                        "classifications"],...
        }
        """

        # A dictionary that stores of all entriess with their tid.
        # It is a key-value map of {tid: entry data in list format}.
        # e.g., {1423543453: [type, id, begin, end, attr_1, ..., attr_n]}
        self.entry_dict: dict = {}

    @abstractmethod
    def add_annotation_raw(self, type_id: int, begin: int, end: int) -> int:
        r"""This function adds an annotation entry with `begin` and `end` index
        to the sortedlist at index `type_id` of the array which records all
        sortedlists, return tid for the entry.

        Args:
            type_id (int): The index of Annotation sortedlist in the array.
            begin (int): begin index of the entry.
            end (int): end index of the entry.

        Returns:
            The tid of the entry.

        """
        # We create an unique tid for the entry. We create the entry list with
        # the format [entry_type, tid, begin, end, none, ...] with all fields
        # filled. We add the entry list to the `entry_type` sortedlist.
        # We add {tid: entry list} to the entry_dict.
        raise NotImplementedError

    @abstractmethod
    def set_attr(self, tid: int, attr_name: str, attr_value):
        r"""This function locates the entry list with tid and sets its
        attribute `attr_name` with value `attr_value`.

        Args:
            tid (int): Unique id of the entry.
            attr_name (str): name of the attribute.
            attr_value: value of the attribute.

        """
        # We retrieve the entry list from entry_dict using tid. We get its
        # entry type. We then locate the index of the attribute using the entry
        # type, field dictionary and attr_name, and update the attribute.

        raise NotImplementedError

    @abstractmethod
    def get_attr(self, tid: int, attr_name: str):
        r"""This function locates the entry list with tid and gets the value
        of `attr_name` of this entry.

        Args:
            tid (int): Unique id of the entry.
            attr_name (str): name of the attribute.

        Returns:

        """
        # We retrieve the entry list from entry_dict using tid. We get its
        # entry type. We then locate the index of the attribute using the entry
        # type, field dictionary and attr_name, and get the attribute.

        raise NotImplementedError

    @abstractmethod
    def delete_entry(self, tid: int):
        r"""This function locates the entry list with tid and removes it
        from the data store. It removes the entry list from both entry_dict
        and the sortedlist of its type.

        Args:
            tid (int): Unique id of the entry.

        """
        # We retrieve the entry list from entry_dict using tid. We get its
        # entry type, type id, begin and end indexes. Then, we find the
        # `entry_type` sortedlist using type id. We bisect the sortedlist
        # to find the entry list. We then remove the entry list from both
        # entry_dict and the `entry_type` sortedlist.

        raise NotImplementedError

    @abstractmethod
    def get_entry(self, tid: int) -> List:
        r"""Look up the entry_dict with key `tid`.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The entry which tid corresponds to.

        """
        raise NotImplementedError

    @abstractmethod
    def get(
        self, entry_type: Union[str, Type[EntryType]], **kwargs
    ) -> Iterator[List]:
        """
        Implementation of this method should provide to obtain the entries of
        type entry_type.

        Args:
            entry_type: The type of the entry to obtain.

        Returns:
            An iterator of the entries matching the provided arguments.

        """

        # We find the type id according to entry_type and locate sortedlist.
        # We create an iterator to generate entries from the sortedlist.

        raise NotImplementedError

    @abstractmethod
    def next_entry(self, tid):
        r"""Get the next entry of the same type as the tid entry.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The next entry of the same type as the tid entry.

        """
        # We retrieve the entry list from entry_dict using tid. We get its
        # entry type, type id, begin and end indexes. Then, we find the
        # `entry_type` sortedlist using type id. We bisect the sortedlist
        # to find the next same type entry.

        raise NotImplementedError

    @abstractmethod
    def prev_entry(self, tid):
        r"""Get the previous entry of the same type as the tid entry.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The previous entry of the same type as the tid entry.

        """
        # We retrieve the entry list from entry_dict using tid. We get its
        # entry type, type id, begin and end indexes. Then, we find the
        # `entry_type` sortedlist using type id. We bisect the sortedlist
        # to find the next same type entry.

        raise NotImplementedError
