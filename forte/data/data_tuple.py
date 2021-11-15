from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Type,
    Union,
    Any,
    Set,
    Callable,
    Tuple,
)

import logging
from collections import defaultdict
import uuid
from sortedcontainers import SortedList
from forte.data.ontology.core import Entry, FDict, FList
from forte.data.ontology.core import EntryType
from forte.data.ontology.top import (
    Annotation,
    Link,
    Group,
    SinglePackEntries,
    Generics,
)
from forte.data import data_utils_io
from forte.data.types import DataRequest
from forte.utils import create_class_with_kwargs
from forte.data.data_pack import as_entry_type, get_class
from forte.data.base_data_structure import BaseDataStructure
from forte.data.types import ReplaceOperationsType, DataRequest


logger = logging.getLogger(__name__)

def typeof(tuple):
    return tuple[0]
    
def begin(tuple):
    return tuple[1]

def end(tuple):
    return tuple[2]

def tid(tuple):
    return tuple[3]

class DataTuple(BaseDataStructure):
    def __init__(self, pack_name: Optional[str] = None):
        super().__init__()
        self._text = ""

        # anntations: list of (class_name, begin, end, args*[tuple])
        self.elements: SortedList[tuple] = SortedList(key = lambda x: (x[1], x[2]))
        self.entry_dict: dict = dict()

    def __iter__(self):
        yield from self.elements
    
    def _validate(self, entry) -> bool:
        return isinstance(entry, tuple)

    def get_span_text(self, begin: int, end: int) -> str:
        r"""Get the text in the data pack contained in the span.

        Args:
            begin (int): begin index to query.
            end (int): end index to query.

        Returns:
            The text within this span.
        """
        return self._text[begin:end]

    def set_text(
        self,
        text: str,
        replace_func: Optional[Callable[[str], ReplaceOperationsType]] = None,
    ):

        if len(text) < len(self._text):
            raise ProcessExecutionException(
                "The new text is overwriting the original one with shorter "
                "length, which might cause unexpected behavior."
            )

        if len(self._text):
            logging.warning(
                "Need to be cautious when changing the text of a "
                "data pack, existing entries may get affected. "
            )

        span_ops = [] if replace_func is None else replace_func(text)

        # The spans should be mutually exclusive
        (
            self._text,
            self.__replace_back_operations,
            self.__processed_original_spans,
            self.__orig_text_len,
        ) = data_utils_io.modify_text_and_track_ops(text, span_ops)

    def get_original_text(self):
        r"""Get original unmodified text from the :class:`DataPack` object.

        Returns:
            Original text after applying the `replace_back_operations` of
            :class:`DataPack` object to the modified text
        """
        original_text, _, _, _ = data_utils_io.modify_text_and_track_ops(
            self._text, self.__replace_back_operations
        )
        return original_text

    """
    New methods for tuple-based opertaions
    """
    def text(self, entry: tuple) -> str:
        return self.get_span_text(begin(entry), end(entry))

    def get_entry(self, tid: int):
        entry = self.entry_dict[tid]
        if entry is None:
            raise KeyError(
                f"There is no entry with tid '{tid}'' in this datapack"
            )
        return entry

    def get_raw(
        self,
        entry_type: Union[str, Type[EntryType]],
        range_annotation: Union[int, tuple] = None,
        include_sub_type=True,
    ):
        entry_type_: Type[EntryType] = as_entry_type(entry_type)
        
        range_annotation_: Tuple

        if isinstance(range_annotation, int):
            range_annotation_ = self.entry_dict[range_annotation]
        else:
            range_annotation_ = range_annotation

        if len(self.elements) == 0 and range_annotation_ is not None:
            yield from []
            return

        if not issubclass(entry_type_, Annotation):
            # temporarily only support get raw for annotation type
            raise ValueError(
                    f"The requested type {str(entry_type_)} is not supported."
                )
        
        all_types: Set[Type]
        if include_sub_type: # not supported currently
            all_types = self._expand_to_sub_types(entry_type_)
        else:
            all_types = {entry_type_}

        entry_iter: Iterator[Entry]
        if range_annotation_ is not None:
            entry_iter = self.iter_in_range(entry_type_, range_annotation_)
        else:
            entry_iter = self.elements

        for entry in entry_iter:
            # Filter by type and components.
            if entry[0] not in all_types:
                continue
            yield entry  # type: ignore
    
    def get_data_raw(
        self,
        context_type: Union[str, Type[Annotation]],
        request: Optional[DataRequest] = None,
        skip_k: int = 0,
    ) -> Iterator[Dict[str, Any]]:
        r"""Fetch entries from the data_pack of type `context_type`.

        Currently, we do not support Groups and Generics in the request.

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
                pack.get_data(base_ontology.zf, requests)

        Args:
            context_type (str): The granularity of the data context, which
                could be any ``Annotation`` type.
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
        context_type_: Type[Annotation]
        if isinstance(context_type, str):
            context_type_ = get_class(context_type)
            if not issubclass(context_type_, Entry):
                raise ValueError(
                    f"The provided `context_type` [{context_type_}] "
                    f"is not a subclass to the"
                    f"`forte.data.ontology.top.Annotation` class"
                )
        else:
            context_type_ = context_type

        annotation_types: Dict[Type[Annotation], Union[Dict, List]] = {}
        link_types: Dict[Type[Link], Union[Dict, List]] = {}
        group_types: Dict[Type[Group], Union[Dict, List]] = {}
        generics_types: Dict[Type[Generics], Union[Dict, List]] = {}

        if request is not None:
            for key_, value in request.items():
                key = as_entry_type(key_)
                if issubclass(key, Annotation):
                    annotation_types[key] = value
                else:
                    raise ValueError(
                        r"Entry type other than Annotation is not support by raw currently"
                    )

        context_args = annotation_types.get(context_type_)

        context_components, _, context_fields = self._parse_request_args(
            context_type_, context_args
        )

        skipped = 0
        # must iterate through a copy here because self.annotations is changing
        # `context` is now a tuple!
        for context in list(self.elements):
            if context[0] != context_type_:
                continue
            if skipped < skip_k:
                skipped += 1
                continue

            data: Dict[str, Any] = {}
            data["context"] = self._text[context[1] : context[2]]
            data["offset"] = context[1]

            for field in context_fields:
                data[field] = self.get_attr_from_tuple(context, field)

            if annotation_types:
                for a_type, a_args in annotation_types.items():
                    if issubclass(a_type, context_type_):
                        continue
                    if a_type.__name__ in data.keys():
                        raise KeyError(
                            f"Requesting two types of entries with the "
                            f"same class name {a_type.__name__} at the "
                            f"same time is not allowed"
                        )
                    data[
                        a_type.__name__
                    ] = self._generate_annotation_entry_data(
                        a_type, a_args, data, context
                    )

            if link_types:
                raise NotImplementedError(
                    "Querying links based on ranges is "
                    "currently not supported."
                )

            # TODO: Getting Group based on range is not done yet.
            if group_types:
                raise NotImplementedError(
                    "Querying groups based on ranges is "
                    "currently not supported."
                )

            if generics_types:
                raise NotImplementedError(
                    "Querying generic types based on ranges is "
                    "currently not supported."
                )

            yield data

    def add_entry_raw(self,
        entry_type, begin, end,
        component_name: Optional[str] = None
    ) -> int:
        # add an entry and return a unique id for it

        tid: int = uuid.uuid4().int
        entry_tuple = [entry_type, begin, end, tid]

        self.elements.add(entry_tuple)
        self.entry_dict[tid] = entry_tuple
        return tid

    def set_attr(self, tid, attr_name, attr_value):
        # check if it exists
        entry_tuple = self.entry_dict[tid]
        entry_tuple.append((attr_name, attr_value))

    def get_attr(self, tid, attr_name) -> List:
        entry = self.entry_dict[tid]
        return self.get_attr_from_tuple(entry, attr_name)

    def get_attr_from_tuple(self, entry: tuple, attr_name: str):
        for attr, val in entry[4:]:
            if attr == attr_name:
                return val
        return None

    def delete_entry(self, tid):
        target = self.elements
        entry_tuple = self.entry_dict[tid]
        tid = entry_tuple[4]

        begin: int = target.bisect_left(entry_tuple)

        index_to_remove = -1
        for i, e in enumerate(target[begin:]):
            if e[3] == tid:
                index_to_remove = begin + i
                break

        if index_to_remove < 0:
            logger.warning(
                "The entry with id %d that you are trying to removed "
                "does not exists in the data pack's index. Probably it is "
                "created but not added in the first place.",
                tid,
            )
        else:
            target.pop(index_to_remove)

    """
    helper functions
    """
    def _get_attributes(self, entry: EntryType) -> List:
        attributes = []
        for attr, value in entry.__dict__.items():
            if attr == "_Entry__pack" or attr == "_begin" or attr == "_end":
                continue
            attributes.append((attr, value))
        return attributes

    def _entry_to_tuple(self, entry: EntryType) -> Tuple:
        """ 
            turn an entry class into a tuple
            entry is a class of EntryType
        """
        fields = [type(entry), entry.begin, entry.end, entry._tid]
        attrs = self._get_attributes(entry)
        entry_tuple = tuple(fields + attrs)
        return entry_tuple
    
    def _create_entry_with_tuple(self, entry):
        type_name = str(entry[0])
        class_args_dict = {
            "pack": self,
            "begin": entry[1],
            "end": entry[2]
            }
        
        attributes_dict = dict()
        for i in range(4, len(entry)):
            attr_tuple = entry[i]
            if isinstance(attr_tuple[1], tuple): # attribute value is also a class
                nested_class = self._create_entry_with_tuple(attr_tuple[1])
                attributes_dict[attr_tuple[0]] = nested_class
            elif isinstance(attr_tuple[1], FList): # turn Flist into a normal list
                list = []
                for element in attr_tuple[1]:
                    if isinstance(element, tuple):
                        nested_class = self._create_entry_with_tuple(element)
                        list.append(nested_class)
                    else:
                        list.append(element)
                attributes_dict[attr_tuple[0]] = list
            elif isinstance(attr_tuple[1], FDict): # turn FDict into a normal dictionary
                dic = dict()
                for key, value in attr_tuple[1].items():
                    if isinstance(value, tuple):
                        nested_class = self._create_entry_with_tuple(element)
                        dic[key] = value
                    else:
                        dic[key] = value
                attributes_dict[attr_tuple[0]] = dic
            else:
                attributes_dict[attr_tuple[0]] = attr_tuple[1]
                    
        klass = create_class_with_kwargs(type_name, class_args_dict)
        klass.__dict__.update(attributes_dict)
        return klass

    def iter_in_range(
        self, entry_type: Type[EntryType], range_annotation: tuple
    ) -> Iterator[EntryType]:
        """
        Iterate the entries of the provided type within or fulfill the
        constraints of the `range_annotation`. The constraint is True if
        an entry is `in_span` of the provided `range_annotation`.

        Internally, if the coverage index between the entry type and the
        type of the `range_annotation` is built, then this will create the
        iterator from the index. Otherwise, the function will iterate them
        from scratch (which is slower). If there are frequent usage of this
        function, it is suggested to build the coverage index.

        Args:
            entry_type: The type of entry to iterate over.
            range_annotation: The range annotation that serve as the constraint.

        Returns:
            An iterator of the entries with in the `range_annotation`.

        """

        if issubclass(entry_type, Annotation):
            range_begin = begin(range_annotation) if range_annotation else 0
            range_end = (
                end(range_annotation)
                if range_annotation
                else end(self.elements[-1])
            )

            # if issubclass(entry_type, Annotation):
            temp_begin = (entry_type, range_begin, range_begin, 0)
            begin_index = self.elements.bisect(temp_begin)

            temp_end = (entry_type, range_end, range_end, 0)
            end_index = self.elements.bisect(temp_end)

            # Make sure these temporary annotations are not part of the
            # actual data.
           
            yield from self.elements[begin_index:end_index]
        else:
            raise ValueError (
                f"only support annotation type"
            )

    def _parse_request_args(self, a_type, a_args):
        # request which fields generated by which component
        components = None
        unit = None
        fields = set()
        if isinstance(a_args, dict):
            components = a_args.get("component")
            # pylint: disable=isinstance-second-argument-not-valid-type
            # TODO: until fix: https://github.com/PyCQA/pylint/issues/3507
            if components is not None and not isinstance(components, Iterable):
                raise TypeError(
                    "Invalid request format for 'components'. "
                    "The value of 'components' should be of an iterable type."
                )
            unit = a_args.get("unit")
            if unit is not None and not isinstance(unit, str):
                raise TypeError(
                    "Invalid request format for 'unit'. "
                    "The value of 'unit' should be a string."
                )
            a_args = a_args.get("fields", set())

        # pylint: disable=isinstance-second-argument-not-valid-type
        # TODO: disable until fix: https://github.com/PyCQA/pylint/issues/3507
        if isinstance(a_args, Iterable):
            fields = set(a_args)
        elif a_args is not None:
            raise TypeError(
                f"Invalid request format for '{a_type}'. "
                f"The request should be of an iterable type or a dict."
            )

        fields.add("tid")
        return components, unit, fields


    def __add_entry_for_annot(
        self, entry: EntryType, allow_duplicate: bool = True
    ) -> EntryType:
        r"""Internal method to add an :class:`~forte.data.ontology.core.Entry`
        object to the :class:`~forte.data.DataPack` object.
        For new Datapack data structure

        Args:
            entry (Entry): An :class:`Entry` object to be added to the datapack.
            allow_duplicate (bool): Whether we allow duplicate in the datapack.

        Returns:
            The input entry itself
        """
        target = self.elements

        begin, end = entry.begin, entry.end

        if begin < 0:
            raise ValueError(
                f"The begin {begin} is smaller than 0, this"
                f"is not a valid begin."
            )

        if end > len(self.text):
            if len(self.text) == 0:
                raise ValueError(
                    f"The end {end} of span is greater than the text "
                    f"length {len(self.text)}, which is invalid. The text "
                    f"length is 0, so it may be the case the you haven't "
                    f"set text for the data pack. Please set the text "
                    f"before calling `add_entry` on the annotations."
                )
            else:
                pack_ref = entry.pack.pack_id
                raise ValueError(
                    f"The end {end} of span is greater than the text "
                    f"length {len(self.text)}, which is invalid. The "
                    f"problematic entry is of type {entry.__class__} "
                    f"at [{begin}:{end}], in pack {pack_ref}."
                )
        # add annotation to a list of tuples
        entry_tuple = self._entry_to_tuple(entry)
        target.add(entry_tuple)
            
        return entry

    
