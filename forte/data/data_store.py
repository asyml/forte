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

from typing import List, Iterator, Tuple, Optional, Any
import uuid
from bisect import bisect_left
from heapq import heappush, heappop
from forte.utils import get_class
from forte.data.base_store import BaseStore
from forte.data.entry_type_generator import EntryTypeGenerator
from forte.data.ontology.top import Annotation
from forte.common import constants

__all__ = ["DataStore"]


class DataStore(BaseStore):
    # TODO: temporarily disable this for development purposes.
    # pylint: disable=pointless-string-statement

    def __init__(self, onto_file_path: Optional[str] = None):
        r"""An implementation of the data store object that mainly uses
        primitive types. This class will be used as the internal data
        representation behind data pack. The usage of primitive types provides
        a speed-up to the previous class-based solution.

        A DataStore object uses primitive types and simple python data
        structures to store a collection of Forte entries for certain types of
        unstructured data.
        Currently, DataStore supports storing data structures with linear span
        (e.g. Annotation), and relational data structures (e.g Link and Group).
        Future extension of the class may support data structures with 2-d range
         (e.g. bounding boxes).

        Internally, we store every entry in a variable ``__elements``, which is
        a nested list: a list of ``entry lists``.

        Every inner list, the ``entry list``, is a list storing entries for a
        single particular type, such as entries for
        :class:`~ft.onto.base_ontology.Sentence`. Different types are stored in
        different lists: [ <Document List>, <Sentence List>, ...]. We will
        discuss the sorting order later.

        The outer list, stores a list of ``entry lists``,
        and each ``entry list``
        is indexed by the type of its element. Specifically, each type is
        associated with a unique ``type_id``, which is generated by the system.
        The mapping between ``type_name`` and ``type_id`` is defined by a
        dictionary ``self.__type_index_dict``.

        Entry information is stored as ``entry data`` in each ``entry list``.
        Each element in the ``entry list`` (an entry data) corresponds to one
        entry instance.

        Each ``entry data`` in the ``entry list`` is represented by a list of
        attributes.
        For example, an annotation type entry has the following format:
        [<begin>, <end>, <tid>, <type_name>, <attr_1>, <attr_2>, ...,
        <attr_n>].
        A group type entry has the following format:
        [<member_type>, <[members_tid_list]>, <tid>, <type_name>, <attr_1>,
            <attr_2>, ..., <attr_n>, index_id].
        A link type entry has the following format:
        [<parent_tid>, <child_tid>, <tid>, <type_name>, <attr_1>, <attr_2>,
        ..., <attr_n>, index_id].

        The first four fields are compulsory for every ``entry data``. The third
        and fourth fields are always ``tid`` and ``type_name``, but the first and
        second fields can change across different types of entries.
        For example, first four fields of Annotation-Like (e.g. subclasses of
        Annotation or AudioAnnotation) entries are always in the order of
        ``begin``, ``end``, ``tid`` and ``type_name``. ``begin`` and ``end``, which are
        compulsory for annotations entries, represent the begin and end
        character indices of entries in the payload.

        The last field is always ``index_id`` for entries that are not
        Annotation-like. It is an extra field to record the location of the
        entry in the list. When the user add a new entry to the data store,
        the ``index_id`` will be created and appended to the end of the original
        ``entry data`` list.

        Here, ``type_name`` is the fully qualifie name of this type represented
        by ``entry list``. It must be a valid ontology defined as a class.
        ``tid`` is a unique id of every entry, which is internally generated by
        uuid.uuid4().
        Each ``type_name`` corresponds to a pre-defined ordered list of
        attributes, the exact order is determined by the system through the
        ontology specifications.
        E.g. an annotation-type ``entry data`` with type
        :class:`~ft.onto.base_ontology.Document` has the following structure:
        [<begin>, <end>, <tid>, <type_name>, <document_class>, <sentiment>,
        <classifications>].
        Here, <document_class>, <sentiment>, <classifications> are the 3
        attributes of this type. This allows the ``entry list`` behaves like a
        table, we can find the value of an attribute through the correct
        ``index_id`` id (e.g. index of the outer list) and `attr_id`
        (e.g. index of the inner list).

        Note that, if the type of ``entry list`` is Annotation-Like (e.g.
        subclasses of Annotation or AudioAnnotation), these entries will be
        sorted by the first two attributes (``begin``, ``end``). However, the
        order of a list with types that are not Annotation-like, is currently
        based on the insertion order.

        ``onto_file_path`` is an optional argument, which allows one to pass in
        a user defined ontology file. This will enable the DataStore to
        understand and store ``entry_type`` defined in the provided file.

        Args:
            onto_file_path (str, optional): the path to the ontology file.
        """
        super().__init__()
        self.onto_file_path = onto_file_path

        """
        The ``_type_attributes`` is a private dictionary that provides
        ``type_name`` and the order of corresponding attributes except ``index_id``.
        The outer keys are indices of ``entry lists`` in ``self.__elements`` as
        integers, representing all types that inherit the ``Entry`` class.
        The inner keys are all the valid attributes for this type.
        The values are the indices of attributes among these lists.

        This structure is supposed to be obtained by processing the dictionary
        generated by another function get_type_attributes(). This function will
        be called in class ``EntryTypeGenerator``.

        Example:

        .. code-block:: python

            self._type_attributes = EntryTypeGenerator.get_type_attributes()

            # self._type_attributes is:
            # {
            #     "ft.onto.base_ontology.Token": {"pos": 4, "ud_xpos": 5,
            #               "lemma": 6, "chunk": 7, "ner": 8, "sense": 9,
            #               "is_root": 10, "ud_features": 11, "ud_misc": 12},
            #     "ft.onto.base_ontology.Document": {"document_class": 4,
            #               "sentiment": 5, "classifications": 6},
            #     "ft.onto.base_ontology.Sentence": {"speaker": 4,
            #               "part_id": 5, "sentiment": 6,
            #               "classification": 7, "classifications": 8},
            # }
        """
        # Issue #570 implements get_type_attributes()
        # see https://github.com/asyml/forte/issues/570
        self._type_attributes: dict = EntryTypeGenerator.get_type_attributes()

        """
        The `__elements` is an underlying storage structure for all the entry
        data added by users in this DataStore class.
        It is a dict of {str: list} pairs that stores sorted ``entry lists`` by
         ``type_name``s.

            Example:
            self.__elements = {
                "ft.onto.base_ontology.Token": Token SortedList(),
                "ft.onto.base_ontology.Document": Document SortedList(),
                "ft.onto.base_ontology.Sentence": Sentence SortedList(),
                ...
            }
        """
        self.__elements: dict = {}

        """
        A dictionary that keeps record of all entrys with their tid.
        It is a key-value map of {tid: entry data in list format}.

        e.g., {1423543453: [begin, end, tid, type_name, attr_1, ..., attr_n],
        4345314235: [parent_tid, child_tid, tid, type_name, attr_1, ...,
                    attr_n, index_id]}
        """
        self.__entry_dict: dict = {}

    def _new_tid(self) -> int:
        r"""This function generates a new ``tid`` for an entry."""
        return uuid.uuid4().int

    def _new_annotation(self, type_name: str, begin: int, end: int) -> List:
        r"""This function generates a new annotation with default fields.
        All default fields are filled with None.
        Called by add_annotation_raw() to create a new annotation with
        ``type_name``, ``begin``, and ``end``.

        Args:
            type_name (str): The fully qualified type name of the new entry.
            begin (int): Begin index of the entry.
            end (int): End index of the entry.

        Returns:
            A list representing a new annotation type entry data.
        """

        tid: int = self._new_tid()
        entry: List[Any]
        entry = [begin, end, tid, type_name]
        entry += len(self._type_attributes[type_name]) * [None]
        return entry

    def _new_link(
        self, type_name: str, parent_tid: int, child_tid: int
    ) -> List:
        r"""This function generates a new link with default fields. All
        default fields are filled with None.
        Called by add_link_raw() to create a new link with ``type_name``,
        ``parent_tid``, and ``child_tid``.

        Args:
            type_name (str): The fully qualified type name of the new entry.
            parent_tid (int): ``tid`` of the parent entry.
            child_tid (int): ``tid`` of the child entry.

        Returns:
            A list representing a new link type entry data.
        """

        tid: int = self._new_tid()
        entry: List[Any]
        entry = [parent_tid, child_tid, tid, type_name]
        entry += len(self._type_attributes[type_name]) * [None]
        return entry

    def _new_group(self, type_name: str, member_type: str) -> List:
        r"""This function generates a new group with default fields. All
        default fields are filled with None.
        Called by add_group_raw() to create a new group with
        ``type_name`` and ``member_type``.

        Args:
            type_name (str): The fully qualified type name of the new entry.
            member_type (str): Fully qualified name of its members.

        Returns:
            A list representing a new group type entry data.
        """

        tid: int = self._new_tid()
        entry = [member_type, [], tid, type_name]
        entry += len(self._type_attributes[type_name]) * [None]
        return entry

    def _is_annotation(self, type_name: str) -> bool:
        r"""This function takes a type_id and returns whether a type
        is an annotation type or not.

        Args:
            type_name (str): The name of type in `self.__elements`.

        Returns:
            A boolean value whether this type_id belongs to an annotation
            type or not.
        """
        entry_class = get_class(type_name)
        return issubclass(entry_class, Annotation)

    def add_annotation_raw(self, type_name: str, begin: int, end: int) -> int:
        r"""This function adds an annotation entry with ``begin`` and ``end``
        indices to current data store object. Returns the ``tid`` for the inserted
        entry.

        Args:
            type_name (str): The fully qualified type name of the new Annotation.
            begin (int): Begin index of the entry.
            end (int): End index of the entry.

        Returns:
            ``tid`` of the entry.
        """
        # We should create the `entry data` with the format
        # [begin, end, tid, type_id, None, ...].
        # A helper function _new_annotation() can be used to generate a
        # annotation type entry data with default fields.
        # A reference to the entry should be store in both self.__elements and
        # self.__entry_dict.
        raise NotImplementedError

    def add_link_raw(
        self, type_name: str, parent_tid: int, child_tid: int
    ) -> Tuple[int, int]:
        r"""This function adds a link entry with ``parent_tid`` and ``child_tid``
        to current data store object. Returns the ``tid`` and the ``index_id`` for
        the inserted entry in the list. This ``index_id`` is the index of the entry
        in the ``type_name`` list.

        Args:
            type_name (str):  The fully qualified type name of the new Link.
            parent_tid (int): ``tid`` of the parent entry.
            child_tid (int): ``tid`` of the child entry.

        Returns:
            ``tid`` of the entry and its index in the ``type_name`` list.

        """
        raise NotImplementedError

    def add_group_raw(
        self, type_name: str, member_type: str
    ) -> Tuple[int, int]:
        r"""This function adds a group entry with ``member_type`` to the
        current data store object. Returns the ``tid`` and the ``index_id``
        for the inserted entry in the list. This ``index_id`` is the index
        of the entry in the ``type_name`` list.

        Args:
            type_name (str): The fully qualified type name of the new Group.
            member_type (str): Fully qualified name of its members.

        Returns:
            ``tid`` of the entry and its index in the (``type_id``)th list.

        """
        raise NotImplementedError

    def set_attribute(self, tid: int, attr_name: str, attr_value: Any):
        r"""This function locates the entry data with ``tid`` and sets its
        ``attr_name`` with `attr_value`. It first finds ``attr_id``  according
        to ``attr_name``. ``tid``, ``attr_id``, and ``attr_value`` are
        passed to `set_attr()`.

        Args:
            tid (int): Unique Id of the entry.
            attr_name (str): Name of the attribute.
            attr_value (any): Value of the attribute.

        Raises:
            KeyError: when ``tid`` or ``attr_name`` is not found.
        """
        try:
            entry_type = self.__entry_dict[tid][constants.ENTRY_TYPE_INDEX]
        except KeyError as e:
            raise KeyError(f"Entry with tid {tid} not found.") from e

        try:
            attr_id = self._type_attributes[entry_type][attr_name]
        except KeyError as e:
            raise KeyError(f"{entry_type} has no {attr_name} attribute.") from e

        self._set_attr(tid, attr_id, attr_value)

    def _set_attr(self, tid: int, attr_id: int, attr_value: Any):
        r"""This function locates the entry data with ``tid`` and sets its
        attribute ``attr_id``  with value `attr_value`. Called by
        `set_attribute()`.

        Args:
            tid (int): The unique id of the entry.
            attr_id (int): The id of the attribute.
            attr_value (any): The value of the attribute.
        """
        entry = self.__entry_dict[tid]
        entry[attr_id] = attr_value

    def get_attribute(self, tid: int, attr_name: str) -> Any:
        r"""This function finds the value of ``attr_name`` in entry with
        ``tid``. It locates the entry data with ``tid`` and finds `attr_id`
        of its attribute ``attr_name``. ``tid`` and ``attr_id``  are passed
        to ``get_attr()``.

        Args:
            tid (int): Unique id of the entry.
            attr_name (str): Name of the attribute.

        Returns:
            The value of ``attr_name`` for the entry with ``tid``.

        Raises:
            KeyError: when ``tid`` or ``attr_name`` is not found.
        """
        try:
            entry_type = self.__entry_dict[tid][constants.ENTRY_TYPE_INDEX]
        except KeyError as e:
            raise KeyError(f"Entry with tid {tid} not found.") from e

        try:
            attr_id = self._type_attributes[entry_type][attr_name]
        except KeyError as e:
            raise KeyError(f"{entry_type} has no {attr_name} attribute.") from e

        return self._get_attr(tid, attr_id)

    def _get_attr(self, tid: int, attr_id: int) -> Any:
        r"""This function locates the entry data with ``tid`` and gets the value
        of ``attr_id``  of this entry. Called by `get_attribute()`.

        Args:
            tid (int): Unique id of the entry.
            attr_id (int): The id of the attribute.

        Returns:
            The value of ``attr_id``  for the entry with ``tid``.
        """
        entry = self.__entry_dict[tid]
        return entry[attr_id]

    def delete_entry(self, tid: int):
        r"""This function locates the entry data with ``tid`` and removes it
        from the data store. This function first removes it from `__entry_dict`.

        Args:
            tid (int): Unique id of the entry.

        Raises:
            KeyError: when entry with ``tid`` is not found.
            RuntimeError: when internal storage is inconsistent.
        """
        try:
            # get `entry data` and remove it from entry_dict
            entry_data = self.__entry_dict.pop(tid)
        except KeyError as e:
            raise KeyError(
                f"The specified tid [{tid}] "
                f"does not correspond to an existing entry data "
            ) from e

        _, _, tid, type_name = entry_data[:4]
        try:
            target_list = self.__elements[type_name]
        except KeyError as e:
            raise RuntimeError(
                f"When deleting entry [{tid}], its type [{type_name}]"
                f"does not exist in current entry lists."
            ) from e
        # complexity: O(lgn)
        # if it's annotation type, use bisect to find the index
        if self._is_annotation(type_name):
            entry_index = bisect_left(target_list, entry_data)
        else:  # if it's group or link, use the index in entry_list
            entry_index = entry_data[-1]

        if (
            entry_index >= len(target_list)
            or target_list[entry_index] != entry_data
        ):
            raise RuntimeError(
                f"When deleting entry [{tid}], entry data is not found in"
                f"the target list of [{type_name}]."
            )

        self._delete_entry_by_loc(type_name, entry_index)

    def _delete_entry_by_loc(self, type_name: str, index_id: int):
        r"""It removes an entry of `index_id` by taking both the `type_id`
        and `index_id`. Called by `delete_entry()`.
        This function will raise an IndexError if the `type_id` or `index_id`
        is invalid.

        Args:
            type_id (int): The index of the list in ``self.__elements``.
            index_id (int): The index of the entry in the list.

        Raises:
            KeyError: when ``type_name`` is not found.
            IndexError: when ``index_id`` is not found.
        """
        try:
            target_list = self.__elements[type_name]
        except KeyError as e:
            raise KeyError(
                f"The specified type [{type_name}] "
                f"does not exist in current entry lists."
            ) from e
        if index_id < 0 or index_id >= len(target_list):
            raise IndexError(
                f"The specified index_id [{index_id}] of type [{type_name}]"
                f"is out of boundary for entry list of length {len(target_list)}."
            )
        target_list.pop(index_id)

    def get_entry(self, tid: int) -> Tuple[List, int, int]:
        r"""This function finds the entry with ``tid``. It returns the entry,
        its ``type_name``, and the index in the ``type_name`` list.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The entry which ``tid`` corresponds to, its ``type_id`` and its index
            in the ``type_name`` list.
        """
        # If the entry is an annotation, bisect the annotation sortedlist
        # to find the entry. May use LRU cache to optimize speed.
        # Otherwise, use ``index_id`` to find the index of the entry.
        raise NotImplementedError

    def co_iterator(self, type_names: List[str]) -> Iterator[List]:
        r"""
        Given two or more type names, iterate their entry lists from beginning to end together.

        For every single type, their entry lists are sorted by the begin and
        end fields. The co_iterator function will iterate those sorted lists
        together, and yield each entry in sorted order. This tasks is quite
        similar to merging several sorted list to one sorted list. We internally
        use a `MinHeap` to order the order of yielded items, and the ordering
        is determined by:

            - start index of the entry.
            - end index of the entry.
            - the index of the entry type name in input parameter ``type_names``.

        The precedence of those values indicates their priority in the min heap
        ordering.
        For example, if two entries have both the same begin and end field,
        then their order is
        decided by the order of user input type_name (the type that first
        appears in the target type list will return first).

        Args:
            type_names (List[str]): a list of string type names

        Returns:

            An iterator of entry elements.

        """

        n = len(type_names)
        # suppose the length of type_names is N and the length of entry list of
        # one type is M
        # then the time complexity of using min-heap to iterate
        # is O(M*log(N))

        # Initialize the first entry of all entry lists
        # it avoids empty entry lists or non-existant entry list
        first_entries = []
        for tn in type_names:
            if tn in self.__elements:
                if len(self.__elements[tn]) > 0:
                    first_entries.append(self.__elements[tn][0])
                else:
                    raise ValueError(
                        f"Entry list of type name ({tn}) is"
                        " empty. Please check data in this DataStore "
                        " to see if empty list is expected"
                        f" or remove {tn} from input parameter type_names"
                    )
            else:
                raise ValueError(
                    f"Input parameter types name {tn} is not"
                    "available. Please input available ones in this DataStore"
                    f"object: {list(self.__elements.keys())}"
                )

        # record the current entry index for elements
        # pointers[i] is the index of entry at (i)th sorted entry lists
        pointers = [0] * n

        # compare tuple (begin, end, order of type name in input argument
        # type_names)
        # we initialize a MinHeap with the first entry of all sorted entry lists
        # in self.__elements
        # the metric of comparing entry order is represented by the tuple
        # (begin index of entry, end index of entry,
        # the index of the entry type name in input parameter ``type_names``)
        h: List[Tuple[Tuple[int, int, int], str]] = []
        for p_idx in range(n):
            entry_tuple = (
                (
                    first_entries[p_idx][constants.BEGIN_INDEX],
                    first_entries[p_idx][constants.END_INDEX],
                    p_idx,
                ),
                first_entries[p_idx][constants.ENTRY_TYPE_INDEX],
            )
            heappush(
                h,
                entry_tuple,
            )

        while h:
            # NOTE: we push the tuple to the heap
            # but not the actual entry. But we can retrieve
            # the entry by the tuple's data.
            # In the following comments,
            # the current entry means the entry that
            # the popped tuples represents
            # the current entry list means the entry list
            # where the current entry locates at.

            # retrieve the popped entry tuple (minimum item in the heap)
            # and get the p_idx (the index of the current entry list in self.__elements)
            entry_tuple = heappop(h)
            type_name: str = entry_tuple[1]
            _, _, p_idx = entry_tuple[0]
            # get the index of current entry
            # and locate the entry represented by the tuple for yielding
            pointer = pointers[p_idx]
            entry = self.__elements[type_name][pointer]
            # check whether there is next entry in the current entry list
            # if there is, then we push the new entry's tuple into the heap
            if pointer + 1 < len(self.__elements[type_name]):
                pointers[p_idx] = pointer + 1
                new_pointer = pointers[p_idx]
                new_entry = self.__elements[type_name][new_pointer]
                new_entry_tuple = (
                    (
                        new_entry[constants.BEGIN_INDEX],
                        new_entry[constants.END_INDEX],
                        p_idx,
                    ),
                    new_entry[constants.ENTRY_TYPE_INDEX],
                )
                heappush(
                    h,
                    new_entry_tuple,
                )
            yield entry

    def get(
        self, type_name: str, include_sub_type: bool = True
    ) -> Iterator[List]:
        r"""This function fetches entries from the data store of
        type ``type_name``.

        Args:
            type_name (str): The fully qualified name of the entry.
            include_sub_type: A boolean to indicate whether get its subclass.

        Returns:
            An iterator of the entries matching the provided arguments.
        """
        if include_sub_type:
            entry_class = get_class(type_name)
            all_types = []
            # iterate all classes to find subclasses
            for type in self.__elements:
                if issubclass(get_class(type), entry_class):
                    all_types.append(type)
            for type in all_types:
                for entry in self.__elements[type]:
                    yield entry
        else:
            try:
                entries = self.__elements[type_name]
            except KeyError as e:
                raise KeyError(f"type {type_name} does not exist") from e
            for entry in entries:
                yield entry

    def next_entry(self, tid: int) -> List:
        r"""Get the next entry of the same type as the ``tid`` entry.
        Call ``get_entry()`` to find the current index and use it to find
        the next entry.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The next entry of the same type as the ``tid`` entry.
        """
        raise NotImplementedError

    def prev_entry(self, tid: int) -> List:
        r"""Get the previous entry of the same type as the ``tid`` entry.
        Call ``get_entry()`` to find the current index and use it to find
        the previous entry.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The previous entry of the same type as the ``tid`` entry.
        """
        raise NotImplementedError
