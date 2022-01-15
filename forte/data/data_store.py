from typing import Union, Type, List, Iterator, Tuple

import uuid
from forte.data.ontology.core import EntryType
from forte.data.base_store import BaseStore


class DataStore(BaseStore):
    def __init__(self):
        """An implementation from the dataframe-like data store object.

        A DataStore object is used to store a collection of NLP entries in a piece of text.
        We store every entry in a data structure `elements`, which is a list of `entry lists`.
        Every `entry list` is a SortedList, storing the same type of entries.
        For example, subtypes of annotations, including Sentence, Documents, and Phrase,
        are stored in separate SortedLists:
        [ <Document SortedList>, <Sentence SortedList>, ...]
        Different `entry lists` are ordered by the `type id` of the type of entries
        storing in this list.

        Entries are stored as `entry data` in each entry list.
        Each `entry data` in the `entry list` is represented by a list of attributes,
        For example, annotation type entry has the following format:
        [<begin>, <end>, <tid>, <entry_type>, <attr_1>, <attr_2>, ..., <attr_n>].

        The first four fields are compulsory for every `entry data`. They are always in
        the order of `begin`, `end`, `tid` and `entry type`, .
        `Begin` and `end` are the begin and end character index of this entry in the text.
        `Entry type` is the type of this entry. `tid` is a unique id of every entry,
        usually generated by uuid.uuid4().
        Each entry type has a fixed field of attributes.
        E.g. an annotation-type `entry data` with type Document has the following structure:
        [<begin>, <end>, <tid>, <entry_type>, <document_class>, <sentiment>, <classifications>]

        Different entries are sorted by the `begin` attribute of the entry.
        If two entries have the same begin position, then we used the `end` attribute to sort them.

        Args:
            pack_name (Optional[str], optional): A name for this data store.
        """
        super().__init__()

        """
        The `_type_attributes` is a private dictionary that provides entry types
        and the order of corresponding attributes.
        The outer keys are all valid ontology types as strings, including all types
        that inherit `Entry` class. Currently they are all defined in ft.onto and ftx.onto.
        The inner keys are all the valid attributes for this type.
        The values are the index of attributes among sortedlist.

        This structure is supposed to be obtained by processing the dictionary
        generated by another function get_type_attributes(). The helper
        function is attribute_dict_helper().

        Example:

        .. code-block:: python

            type_attributes_list: dict = get_type_attributes()
            self._type_attributes = attribute_dict_helper(type_attributes_list)

            type_attributes_list is:
            {
                "ft.onto.base_ontology.Token": ["pos", "ud_xpos", "lemma",
                          "chunk", "ner", "sense", "is_root", "ud_features",
                          "ud_misc"],
                "ft.onto.base_ontology.Document": ["document_class",
                          "sentiment", "classifications"],
                "ft.onto.base_ontology.Sentence": ["speaker", "part_id",
                          "sentiment", "classification","classifications"],
            }
            self._type_attributes is:
            {
                "ft.onto.base_ontology.Token": {"pos": 4, "ud_xpos": 5,
                          "lemma": 6, "chunk": 7, "ner": 8, "sense": 9,
                          "is_root": 10, "ud_features": 11, "ud_misc": 12},
                "ft.onto.base_ontology.Document": {"document_class": 4,
                          "sentiment": 5, "classifications": 6},
                "ft.onto.base_ontology.Sentence": {"speaker": 4,
                          "part_id": 5, "sentiment": 6, "classification": 7,
                          "classifications": 8},
            }
        """
        # Issue #570 implements get_type_attributes()
        # see https://github.com/asyml/forte/issues/570
        self._type_attributes: dict = {}

        """
        Element is an underlying storage structure for all the entry data added by
        user in this DataStore class.
        It is a list that stores sorted `entry lists` by the order of `type id`.

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
        The `_type_dict` dictionary stores type id, which is the index of
        sortedlists in `self.elements`.
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

    def _new_annotation(self, type: str, begin, end):
        """
        generate a new annotation with default fields. Called by add_entry_raw()
        to create a new annotation with `type`, `begin`, and `end`.

        Args:
            type (str): type of this annotation
        Returns:
            A list representing a new annotation type entry data
        """

        tid: int = uuid.uuid4().int
        entry = [begin, end, tid, type]
        entry += len(self._type_attributes[type]) * [None]
        return entry

    def _generate_attr_index(self):
        """
        For every type in `_type_attributes`, we need to convert the attribute
        list to a dictionary, mapping each attribute string to a unique index.
        The index should be the actual index of this attribute in the entry's
        list. Index 0, 1, 2, 3 should be reserved for begin, end, tid, type.

        For example, for type `ft.onto.base_ontology.Sentence`, the attribute
        ``speaker`` has index 4, and attribute ``part_id`` has index 5.

        """
        raise NotImplementedError

    def add_annotation_raw(
        self, type_id: int, begin: int, end: int
    ) -> Tuple[int, int]:
        r"""This function adds an annotation entry with `begin` and `end` index
        to the sortedlist at index `type_id` of the array which records all
        sortedlists, return tid and index for the entry.

        Args:
            type_id (int): Index of Annotation sortedlist in `self.elements`.
            begin (int): begin index of the entry.
            end (int): end index of the entry.

        Returns:
            `tid` of the entry and its index in the entry list.
        """
        # We should create the `entry data` with the format
        # [begin, end, tid, entry_type, None, ...].
        # A helper function _new_annotation() can be used to generate a annotation
        # type entry data with default fields.
        # A reference to the entry should be store in both self.elements and
        # self.entry_dict.
        raise NotImplementedError

    def set_attr(self, tid: int, attr_id: int, attr_value):
        r"""This function locates the entry data with `tid` and sets its
        attribute `attr_id` with value `attr_value`.

        Args:
            tid (int): Unique id of the entry.
            attr_id (int): id of the attribute.
            attr_value: value of the attribute.

        """
        # We retrieve the entry data from entry_dict using tid. We get its
        # entry type. We then locate the attribute using attribute if, and update the attribute.

        raise NotImplementedError

    def get_attr(self, tid: int, attr_id: int):
        r"""This function locates the entry data with `tid` and gets the value
        of `attr_id` of this entry.

        Args:
            tid (int): Unique id of the entry.
            attr_id (int): id of the attribute.

        Returns:
            The value of `attr_id` for the entry with `tid`.
        """
        # We retrieve the entry data from entry_dict using tid. We get its
        # entry type. We then locate the attribute using attr_id, and get the attribute.

        raise NotImplementedError

    def delete_entry(self, tid: int):
        r"""This function locates the entry data with `tid` and removes it
        from the data store. This function first removes it from `entry_dict`.

        Args:
            tid (int): Unique id of the entry.

        """
        # We retrieve the entry data from entry_dict using tid. We get its
        # entry type, `type_id`, begin and end indexes. Remove the entry from
        # `entry_dict` when its information is retrieved. We find the
        # `entry_type` sortedlist using `type_id`. We bisect the target_list to
        # find the index of entry data. Then, `type_id` and index are passed to
        # _delete_entry_by_loc.
        raise NotImplementedError

    def _delete_entry_by_loc(self, type_id: int, index_id: int):
        r"""It removes the index_id'th entry data from the sortedlist of its type.

        Args:
            type_id (int): The index of the sortedlist in `self.elements`.
            index_id (int): Index of the entry in the sortedlist.
        """
        # We then remove the entry data from `entry_type` sortedlist.
        raise NotImplementedError

    def get_entry(self, tid: int) -> Tuple[List, int, int]:
        r"""Look up the entry_dict with key `tid`. Find its type_id and its
        index in the `entry_type` sortedlist.

        Args:
            tid (int): Unique id of the entry.

        Returns:
            The entry which `tid` corresponds to, its type_id and its index
            in the `entry_type` sortedlist.

        """
        raise NotImplementedError

    def get(
        self, entry_type: Union[str, Type[EntryType]], **kwargs
    ) -> Iterator[List]:
        """
        Implementation of this method should provide to obtain the entries of
        type `entry_type`.

        Args:
            entry_type: The type of the entry to obtain.

        Returns:
            An iterator of the entries matching the provided arguments.

        """
        # We find the type id according to entry_type and locate sortedlist.
        # We create an iterator to generate entries from the sortedlist.

        raise NotImplementedError
