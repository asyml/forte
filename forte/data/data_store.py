from typing import Union, Type, List, Iterator, Tuple, Optional, Any

import uuid
from forte.data.ontology.core import EntryType
from forte.data.base_store import BaseStore

__all__ = ["DataStore"]


class DataStore(BaseStore):
    # pylint: disable=pointless-string-statement

    def __init__(self, onto_file_path: Optional[str] = None):
        r"""An implementation of the data store object that mainly uses primitive types.

        A DataStore object is used to store a collection of Forte entries in a piece of text.
        We store every entry in a data structure `elements`, which is a list of `entry lists`.
        Every `entry list` is a SortedList, storing the same type of entries.
        For example, subtypes of annotations, including Sentence, Documents, and Phrase,
        are stored in separate SortedLists:
        [ <Document SortedList>, <Sentence SortedList>, ...]
        Different `entry lists` are ordered by the `type_id` of the type of entries
        stored in this list.

        Entries are stored as `entry data` in each entry list.
        Each `entry data` in the `entry list` is represented by a list of attributes,
        For example, an annotation type entry has the following format:
        [<begin>, <end>, <tid>, <entry_type>, <attr_1>, <attr_2>, ..., <attr_n>].
        A group type entry has the following format:
        [<member_type>, <[members_tid_list]>, <tid>, <entry_type>, <attr_1>,
            <attr_2>, ..., <attr_n>].
        A link type entry has the following format:
        [<parent_tid>, <child_tid>, <tid>, <entry_type>, <attr_1>, <attr_2>, ..., <attr_n>].

        The first four fields are compulsory for every `entry data`. The third
        and fourth fields are always `tid` and `entry_type`, but the first and
        second fields can change across different types of entries.
        For example, first four fields of annotations entries are always in
        the order of `begin`, `end`, `tid` and `entry_type`. `Begin` and `end`,
        which are compulsory for annotations entries, represent the begin and
        end character indices of entries in the text.
        `Entry_type` is the fully qualified name of every entry. `Tid` is an unique id of every entry,
        usually generated by uuid.uuid4().
        Each `entry_type` has a fixed field of attributes.
        E.g. an annotation-type `entry data` with type `ft.onto.base_ontology.Document` has the following structure:
        [<begin>, <end>, <tid>, <entry_type>, <document_class>, <sentiment>, <classifications>].

        Different entries are sorted by the first attribute of the entry.
        For example, the annotation sortedlist uses `begin` to sort entries.

        Args:
            onto_file_path (str, optional): the path to the ontology file.
        """
        super().__init__()
        self.onto_file_path = onto_file_path
        self.tid_idx = 2
        self.entry_type_idx = 3

        """
        The `_type_attributes` is a private dictionary that provides `entry_type`
        and the order of corresponding attributes.
        The outer keys are fully qualified names of valid ontology types as
        strings, including all types that inherit the `Entry` class.
        Currently, they are all defined in ft.onto and ftx.onto.
        The inner keys are all the valid attributes for this type.
        The values are the indices of attributes among sortedlists.

        This structure is supposed to be obtained by processing the dictionary
        generated by another function get_type_attributes(). This function will
        be called in class `EntryTypeGenerator`.

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
            #               "part_id": 5, "sentiment": 6, "classification": 7,
            #               "classifications": 8},
            # }
        """
        # self._type_attributes = EntryTypeGenerator.get_type_attributes()
        # Issue #570 implements get_type_attributes()
        # see https://github.com/asyml/forte/issues/570
        self._type_attributes: dict = {}

        """
        The `elements` is an underlying storage structure for all the entry
        data added by users in this DataStore class.
        It is a list that stores sorted `entry lists` by the order of `type_id`.

            Example:
            self.elements = [
                Token SortedList(),
                Document SortedList(),
                Sentence SortedList(),
                ...
            ]
        """
        self.elements: List = []

        """
        The `_type_dict` is a private dictionary that stores `type_id`, which
        is the index of sortedlists in `self.elements`.
        The keys are fully qualified names of valid ontology types as strings.
        It should be created only once no matter how many data store objects
        are initialized.

        Example:

        .. code-block:: python

            self._type_dict: dict = get_type_id()

            # self._type_dict is:
            # {
            #     "ft.onto.base_ontology.Token": 0,
            #     "ft.onto.base_ontology.Document": 1,
            #     "ft.onto.base_ontology.Sentence": 2,
            # }
        """
        # TODO: implement get_type_id() (Issue #need creation)
        self._type_dict: dict = {}

        """
        A dictionary that keeps record of all entrys with their tid.
        It is a key-value map of {tid: entry data in list format}.

        e.g., {1423543453: [begin, end, tid, type, attr_1, ..., attr_n]}
        """
        self.entry_dict: dict = {}

    def _new_annotation(
        self, entry_type: Union[str, Type[EntryType]], begin: int, end: int
    ):
        r"""This function generates a new annotation with default fields.
        Called by add_annotation_raw() to create a new annotation
        with `type`, `begin`, and `end`.

        Args:
            entry_type (str): Fully qualified name of this annotation.
            begin (int): Begin index of the entry.
            end (int): End index of the entry.

        Returns:
            A list representing a new annotation type entry data.
        """

        tid: int = uuid.uuid4().int
        entry = [begin, end, tid, entry_type]
        entry += len(self._type_attributes[type]) * [None]
        return entry

    def add_annotation_raw(
        self, entry_type: Union[str, Type[EntryType]], begin: int, end: int
    ) -> Tuple[int, int]:
        r"""This function adds an annotation entry with `begin` and `end` indices
        to the `entry_type` sortedlist, returns the tid and the index for the entry.

        Args:
            entry_type (str): Fully qualified name of this annotation.
            begin (int): Begin index of the entry.
            end (int): End index of the entry.

        Returns:
            `Tid` of the entry and its index in the entry list.
        """
        # We should create the `entry data` with the format
        # [begin, end, tid, entry_type, None, ...].
        # A helper function _new_annotation() can be used to generate a annotation
        # type entry data with default fields.
        # A reference to the entry should be store in both self.elements and
        # self.entry_dict.
        raise NotImplementedError

    def set_attr(self, tid: int, attr_id: int, attr_value: Any):
        r"""This function locates the entry data with `tid` and sets its
        attribute `attr_id` with value `attr_value`. Called by `set_attribute()`.

        Args:
            tid (int): Unique id of the entry.
            attr_id (int): Id of the attribute.
            attr_value (any): Value of the attribute.
        """
        # We retrieve the entry data from `entry_dict` using tid.
        # We locate the attribute using `attr_id` and update the attribute.
        entry = self.entry_dict[tid]
        entry[attr_id] = attr_value

    def get_attr(self, tid: int, attr_id: int):
        r"""This function locates the entry data with `tid` and gets the value
        of `attr_id` of this entry. Called by `get_attribute()`.

        Args:
            tid (int): Unique id of the entry.
            attr_id (int): Id of the attribute.

        Returns:
            The value of `attr_id` for the entry with `tid`.
        """
        # We retrieve the entry data from `entry_dict` using tid.
        # We locate the attribute using `attr_id` and get the attribute.
        entry = self.entry_dict[tid]
        return entry[attr_id]

    def set_attribute(self, tid: int, attr_name: str, attr_value: Any):
        r"""This function locates the entry data with `tid` and finds `attr_id`
        of its attribute `attr_name`. `Tid`, `attr_id`, and `attr_value` are
        passed to `set_attr()`.

        Args:
            tid (int): Unique Id of the entry.
            attr_name (str): Name of the attribute.
            attr_value (any): Value of the attribute.
        """
        if tid not in self.entry_dict:
            raise KeyError(f"Entry with tid {tid} not found.")
        entry_type = self.entry_dict[tid][self.entry_type_idx]
        if attr_name not in self._type_attributes[entry_type]:
            raise ValueError(f"{entry_type} has no {attr_name} attribute.")
        attr_id = self._type_attributes[entry_type][attr_name]
        self.set_attr(tid, attr_id, attr_value)

    def get_attribute(self, tid: int, attr_name: str):
        r"""This function locates the entry data with `tid` and finds `attr_id`
        of its attribute `attr_name`. `Tid` and `attr_id` are passed
        to `get_attr()`.

        Args:
            tid (int): Unique id of the entry.
            attr_name (str): Name of the attribute.

        Returns:
            The value of `attr_name` for the entry with `tid`.
        """
        if tid not in self.entry_dict:
            raise KeyError(f"Entry with tid {tid} not found.")
        entry_type = self.entry_dict[tid][self.entry_type_idx]
        if attr_name not in self._type_attributes[entry_type]:
            raise ValueError(f"{entry_type} has no {attr_name} attribute.")
        attr_id = self._type_attributes[entry_type][attr_name]
        return self.get_attr(tid, attr_id)

    def delete_entry(self, tid: int):
        r"""This function locates the entry data with `tid` and removes it
        from the data store. This function first removes it from `entry_dict`.

        Args:
            tid (int): Unique id of the entry.
        """
        # We retrieve the entry data from `entry_dict` using tid. We get its
        # `entry_type`, `type_id`, `begin` and `end` indices. we remove the
        # entry from `entry_dict` when its information is retrieved. We find the
        # `entry_type` sortedlist using `type_id`. We bisect the sortedlist to
        # find the index of entry data. Then, `type_id` and index are passed to
        # _delete_entry_by_loc.
        raise NotImplementedError

    def _delete_entry_by_loc(self, type_id: int, index_id: int):
        r"""It removes the `index_id`'th entry data from the sortedlist of its type.
        Called by `delete_entry()`.

        Args:
            type_id (int): The index of the sortedlist in `self.elements`.
            index_id (int): The index of the entry in the sortedlist.
        """
        # We then remove the entry data from `entry_type` sortedlist.
        raise NotImplementedError

    def get_entry(self, tid: int) -> Tuple[List, int, int]:
        r"""This function looks up the `entry_dict` with key `tid` and finds its
        `type_id` and the index in the `entry_type` sortedlist.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The entry which `tid` corresponds to, its `type_id` and its index
            in the `entry_type` sortedlist.

        """
        raise NotImplementedError

    def get(
        self, entry_type: Union[str, Type[EntryType]], **kwargs
    ) -> Iterator[List]:
        r"""Fetch entries from the data store of type `entry_type`.

        Args:
            entry_type: Fully qualified name of this annotation.

        Returns:
            An iterator of the entries matching the provided arguments.

        """
        # We find the `type_id` according to `entry_type` and locate the sortedlist.
        # We create an iterator to generate entries from the sortedlist.
        yield from self.elements[self._type_dict[entry_type]]
