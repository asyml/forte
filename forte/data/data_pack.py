import logging
from typing import (Dict, Iterable, Iterator, List, Optional, Type, Union, Any,
                    Set, Callable, Tuple)

import numpy as np
from sortedcontainers import SortedList

from forte.common.types import EntryType, ReplaceOperationsType, DataRequest
from forte.data.base_pack import BaseMeta, BasePack
from forte.data.index import BaseIndex
from forte.data.base import Span
from forte.data.ontology.core import Entry
from forte.data.ontology.top import (
    Annotation, Link, Group, SinglePackEntries, Generic, Query)
from forte.data import io_utils

logger = logging.getLogger(__name__)

__all__ = [
    "Meta",
    "DataPack",
]


class Meta(BaseMeta):
    r"""Basic Meta information associated with each instance of
    :class:`~forte.data.data_pack.DataPack`.

    Args:
        doc_id:  An unique ID assigned to identify the data pack.
        language: The language used by this data pack, default is English.
        span_unit: The unit used for interpreting the Span object of this
          data pack. Default is character.
    """

    def __init__(self, doc_id: Optional[str] = None,
                 language: str = 'eng', span_unit: str = 'character'):
        super().__init__(doc_id)
        self.language = language
        self.span_unit = span_unit


class DataPack(BasePack[Entry, Link, Group]):
    # pylint: disable=too-many-public-methods
    r"""A :class:`DataPack` contains a piece of natural language text and a
    collection of NLP entries (annotations, links, and groups). The natural
    language text could be a document617, paragraph or in any other granularity.

    Args:
        doc_id (str, optional): A universal id of this data pack.
    """

    def __init__(self, doc_id: Optional[str] = None):
        super().__init__()
        self._text = ""

        self.annotations: SortedList[Annotation] = SortedList()
        self.links: List[Link] = []
        self.groups: List[Group] = []
        self.generics: List[Generic] = []

        self.replace_back_operations: ReplaceOperationsType = []
        self.processed_original_spans: List[Tuple[Span, Span]] = []
        self.orig_text_len: int = 0

        self.index: DataIndex = DataIndex()
        self.meta: Meta = Meta(doc_id)

    def __getstate__(self):
        r"""
        In serialization,
            1) will serialize the annotation sorted list as a normal list;
            2) will not serialize the indexes
        """
        state = super(DataPack, self).__getstate__()
        state['annotations'] = list(state['annotations'])
        state.pop('index')

        return state

    def __setstate__(self, state):
        r"""
        In deserialization, we
            1) transform the annotation list back to a sorted list;
            2) initialize the indexes.
        """
        super(DataPack, self).__setstate__(state)
        self.annotations = SortedList(self.annotations)
        self.index = DataIndex()
        self.index.update_basic_index(list(self.annotations))
        self.index.update_basic_index(self.links)
        self.index.update_basic_index(self.groups)

        for a in self.annotations:
            a.set_pack(self)

        for a in self.links:
            a.set_pack(self)

        for a in self.groups:
            a.set_pack(self)

        for a in self.generics:
            a.set_pack(self)

    # pylint: disable=no-self-use
    def validate(self, entry: EntryType) -> bool:
        return isinstance(entry, SinglePackEntries)

    @property
    def text(self) -> str:
        r"""Return the text of the data pack"""
        return self._text

    def get_span_text(self, span: Span) -> str:
        r"""Get the text in the data pack contained in the span

        Args:
            span (Span): Span object which contains a `begin` and an `end` index

        Returns:
            The text within this span
        """
        return self._text[span.begin: span.end]

    def set_text(
            self, text: str, replace_func:
            Optional[Callable[[str], ReplaceOperationsType]] = None):

        if len(self._text) > 0:
            logger.warning("The new text is overwriting the original one, "
                           "which might cause unexpected behavior.")

        span_ops = [] if replace_func is None else replace_func(text)

        # The spans should be mutually exclusive
        (
            self._text, self.replace_back_operations,
            self.processed_original_spans, self.orig_text_len
        ) = io_utils.modify_text_and_track_ops(text, span_ops)

    def get_original_text(self):
        r"""Get original unmodified text from the :class:`DataPack` object.

        Returns:
            Original text after applying the `replace_back_operations` of
            :class:`DataPack` object to the modified text
        """
        original_text, _, _, _ = io_utils.modify_text_and_track_ops(
            self._text, self.replace_back_operations)
        return original_text

    def get_original_span(self, input_processed_span: Span,
                          align_mode: str = "relaxed"):
        r"""Function to obtain span of the original text that aligns with the
        given span of the processed text.

        Args:
            input_processed_span: Span of the processed text for which the
            corresponding span of the original text is desired
            align_mode: The strictness criteria for alignment in the ambiguous
            cases, that is, if a part of input_processed_span spans a part
            of the inserted span, then align_mode controls whether to use the
            span fully or ignore it completely according to the following
            possible values

            - "strict" - do not allow ambiguous input, give ValueError
            - "relaxed" - consider spans on both sides
            - "forward" - align looking forward, that is, ignore the span
            towards the left, but consider the span towards the right
            - "backward" - align looking backwards, that is, ignore the span
            towards the right, but consider the span towards the left

        Returns:
            Span of the original text that aligns with input_processed_span

        Example:
            * Let o-up1, o-up2, ... and m-up1, m-up2, ... denote the unprocessed
              spans of the original and modified string respectively. Note that
              each o-up would have a corresponding m-up of the same size.
            * Let o-pr1, o-pr2, ... and m-pr1, m-pr2, ... denote the processed
              spans of the original and modified string respectively. Note that
              each o-p is modified to a corresponding m-pr that may be of a
              different size than o-pr.
            * Original string:
              <--o-up1--> <-o-pr1-> <----o-up2----> <----o-pr2----> <-o-up3->
            * Modified string:
              <--m-up1--> <----m-pr1----> <----m-up2----> <-m-pr2-> <-m-up3->
            * Note that `self.inverse_original_spans` that contains modified
              processed spans and their corresponding original spans, would look
              like - [(o-pr1, m-pr1), (o-pr2, m-pr2)]

            >> data_pack = DataPack()
            >> original_text = "He plays in the park"
            >> data_pack.set_text(original_text,\
            >>                    lambda _: [(Span(0, 2), "She"))]
            >> data_pack.text
            "She plays in the park"
            >> input_processed_span = Span(0, len("She plays"))
            >> orig_span = data_pack.get_original_span(input_processed_span)
            >> data_pack.get_original_text()[orig_span.begin: orig_span.end]
            "He plays"
        """
        assert align_mode in ["relaxed", "strict", "backward", "forward"]

        req_begin = input_processed_span.begin
        req_end = input_processed_span.end

        def get_original_index(input_index: int, is_begin_index: bool,
                               mode: str) -> int:
            r"""
            Args:
                input_index: begin or end index of the input span
                is_begin_index: if the index is the begin index of the input
                span or the end index of the input span
                mode: alignment mode
            Returns:
                Original index that aligns with input_index
            """
            if len(self.processed_original_spans) == 0:
                return input_index

            len_processed_text = len(self._text)
            orig_index = None
            prev_end = 0
            for (inverse_span, original_span) in self.processed_original_spans:
                # check if the input_index lies between one of the unprocessed
                # spans
                if prev_end <= input_index < inverse_span.begin:
                    increment = original_span.begin - inverse_span.begin
                    orig_index = input_index + increment
                # check if the input_index lies between one of the processed
                # spans
                elif inverse_span.begin <= input_index < inverse_span.end:
                    # look backward - backward shift of input_index
                    if is_begin_index and mode in ["backward", "relaxed"]:
                        orig_index = original_span.begin
                    if not is_begin_index and mode == "backward":
                        orig_index = original_span.begin - 1

                    # look forward - forward shift of input_index
                    if is_begin_index and mode == "forward":
                        orig_index = original_span.end
                    if not is_begin_index and mode in ["forward", "relaxed"]:
                        orig_index = original_span.end - 1

                # break if the original index is populated
                if orig_index is not None:
                    break
                prev_end = inverse_span.end

            if orig_index is None:
                # check if the input_index lies between the last unprocessed
                # span
                inverse_span, original_span = self.processed_original_spans[-1]
                if inverse_span.end <= input_index < len_processed_text:
                    increment = original_span.end - inverse_span.end
                    orig_index = input_index + increment
                else:
                    # check if there input_index is not valid given the
                    # alignment mode or lies outside the processed string
                    raise ValueError(f"The input span either does not adhere "
                                     f"to the {align_mode} alignment mode or "
                                     f"lies outside to the processed string.")
            return orig_index

        orig_begin = get_original_index(req_begin, True, align_mode)
        orig_end = get_original_index(req_end - 1, False, align_mode) + 1

        return Span(orig_begin, orig_end)

    def add_entry(self, entry: EntryType) -> EntryType:
        r"""Force add an :class:`~forte.data.ontology.top.Entry` object to the
        :class:`DataPack` object. Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be added to the pack.

        Returns:
            The input entry itself
        """
        return self.__add_entry_with_check(entry, True)

    def add_or_get_entry(self, entry: EntryType) -> EntryType:
        r"""Try to add an :class:`~forte.data.ontology.top.Entry` object to the
        :class:`DataPack` object.

        If the same entry already exists, will return the existing entry
        instead of adding the new one. Note that we regard two entries as the
        same if their :meth:`~forte.data.ontology.top.Entry.eq` have
        the same return value, and users could
        override :meth:`~forte.data.ontology.top.Entry.eq` in their
        custom entry classes.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be added to the pack.

        Returns:
            If a same entry already exists, returns the existing
            entry. Otherwise, return the (input) entry just added.
        """
        return self.__add_entry_with_check(entry, False)

    def __add_entry_with_check(self, entry: EntryType,
                               allow_duplicate: bool = True) -> EntryType:
        r"""Internal method to add an :class:`Entry` object to the
        :class:`DataPack` object.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the datapack.
            allow_duplicate (bool): Whether we allow duplicate in the datapack.

        Returns:
            The input entry itself
        """
        if isinstance(entry, Annotation):
            target = self.annotations
        elif isinstance(entry, Link):
            target = self.links
        elif isinstance(entry, Group):
            target = self.groups
        else:
            target = self.generics
            # raise ValueError(
            #     f"Invalid entry type {type(entry)}. A valid entry "
            #     f"should be an instance of Annotation, Link, or Group."
            # )

        add_new = allow_duplicate or (entry not in target)

        if add_new:
            self.record_entry(entry)

            if isinstance(target, list):
                target.append(entry)
            else:
                # For the sorted list case.
                target.add(entry)

            # update the data pack index if needed
            self.index.update_basic_index([entry])
            if self.index.link_index_on and isinstance(entry, Link):
                self.index.update_link_index([entry])
            if self.index.group_index_on and isinstance(entry, Group):
                self.index.update_group_index([entry])
            self.index.deactivate_coverage_index()
            return entry
        else:
            return target[target.index(entry)]

    def delete_entry(self, entry: EntryType):
        r"""Delete an :class:`~forte.data.ontology.top.Entry` object from the
        :class:`DataPack`.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.top.Entry`
                object to be deleted from the pack.

        """
        begin = 0

        if isinstance(entry, Annotation):
            target = self.annotations
            begin = target.bisect_left(entry)
        elif isinstance(entry, Link):
            target = self.links
        elif isinstance(entry, Group):
            target = self.groups
        else:
            raise ValueError(
                f"Invalid entry type {type(entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )

        for i, e in enumerate(target[begin:]):
            if e.tid == entry.tid:
                target.pop(i + begin)
                break

        # update basic index
        self.index.remove_entry(entry)

        # set other index invalid
        self.index.turn_link_index_switch(on=False)
        self.index.turn_group_index_switch(on=False)
        self.index.deactivate_coverage_index()

    @classmethod
    def validate_link(cls, entry: EntryType) -> bool:
        return isinstance(entry, Link)

    @classmethod
    def validate_group(cls, entry: EntryType) -> bool:
        return isinstance(entry, Group)

    def get_data(self, context_type: Type[Annotation],
                 request: Optional[DataRequest] = None,
                 skip_k: int = 0) -> Iterator[Dict[str, Any]]:
        r"""Fetch entries from the data_pack of type `context_type`.

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
                pack.get_data(base_ontology.Sentence, requests)

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
                data from the `offset` + 1 th instance.

        Returns:
            A data generator, which generates one piece of data (a dict
            containing the required entries, fields, and context).
        """
        annotation_types: Dict[Type[Annotation], Union[Dict, List]] = dict()
        link_types: Dict[Type[Link], Union[Dict, List]] = dict()
        group_types: Dict[Type[Group], Union[Dict, List]] = dict()
        if request is not None:
            for key, value in request.items():
                if issubclass(key, Annotation):
                    annotation_types[key] = value
                elif issubclass(key, Link):
                    link_types[key] = value
                elif issubclass(key, Group):
                    group_types[key] = value

        context_args = annotation_types.get(context_type)

        context_components, _, context_fields = self._parse_request_args(
            context_type, context_args)

        valid_context_ids: Set[int] = self.get_ids_by_type(context_type)
        if context_components:
            valid_component_id: Set[int] = set()
            for component in context_components:
                valid_component_id |= self.get_ids_by_component(component)
            valid_context_ids &= valid_component_id

        skipped = 0
        # must iterate through a copy here because self.annotations is changing
        for context in list(self.annotations):
            if (context.tid not in valid_context_ids or
                    not isinstance(context, context_type)):
                continue
            if skipped < skip_k:
                skipped += 1
                continue

            data: Dict[str, Any] = dict()
            data["context"] = self.text[context.span.begin: context.span.end]
            data["offset"] = context.span.begin

            for field in context_fields:
                data[field] = getattr(context, field)

            if annotation_types:
                for a_type, a_args in annotation_types.items():
                    if issubclass(a_type, context_type):
                        continue
                    if a_type.__name__ in data.keys():
                        raise KeyError(
                            f"Requesting two types of entries with the "
                            f"same class name {a_type.__name__} at the "
                            f"same time is not allowed")
                    data[a_type.__name__] = \
                        self._generate_annotation_entry_data(
                            a_type, a_args, data, context)

            if link_types:
                for l_type, l_args in link_types.items():
                    if l_type.__name__ in data.keys():
                        raise KeyError(
                            f"Requesting two types of entries with the "
                            f"same class name {l_type.__name__} at the "
                            f"same time is not allowed")
                    data[l_type.__name__] = self._generate_link_entry_data(
                        l_type, l_args, data, context)

            if group_types:
                # pylint: disable=unused-variable
                for g_type, g_args in group_types.items():
                    pass

            yield data

    def _parse_request_args(self, a_type, a_args):
        # request which fields generated by which component
        components = None
        unit = None
        fields = set()
        if isinstance(a_args, dict):
            components = a_args.get("component")
            if components is not None and not isinstance(components, Iterable):
                raise TypeError(
                    f"Invalid request format for 'components'. "
                    f"The value of 'components' should be of an iterable type."
                )
            unit = a_args.get("unit")
            if unit is not None and not isinstance(unit, str):
                raise TypeError(
                    f"Invalid request format for 'unit'. "
                    f"The value of 'unit' should be a string."
                )
            a_args = a_args.get("fields", set())

        if isinstance(a_args, Iterable):
            fields = set(a_args)
        elif a_args is not None:
            raise TypeError(
                f"Invalid request format for '{a_type}'. "
                f"The request should be of an iterable type or a dict."
            )

        # # check the existence of fields
        #
        # self.field_records
        #
        # for meta_key, meta_val in self.internal_metas.items():
        #     if issubclass(meta_key, a_type):
        #         for meta_c, meta_f in meta_val.fields_created.items():
        #             if components is None or meta_c in components:
        #                 if not fields.issubset(meta_f):
        #                     raise KeyError(
        #                         f"The {a_type} generated by {meta_c} doesn't "
        #                         f"have the fields requested.")

        fields.add("tid")
        return components, unit, fields

    def _generate_annotation_entry_data(
            self,
            a_type: Type[Annotation],
            a_args: Union[Dict, Iterable],
            data: Dict,
            cont: Optional[Annotation]) -> Dict:

        components, unit, fields = self._parse_request_args(a_type, a_args)

        a_dict: Dict[str, Any] = dict()

        a_dict["span"] = []
        a_dict["text"] = []
        for field in fields:
            a_dict[field] = []

        unit_begin = 0
        if unit is not None:
            if unit not in data.keys():
                raise KeyError(f"{unit} is missing in data. You need to "
                               f"request {unit} before {a_type}.")
            a_dict["unit_span"] = []

        cont_begin = cont.span.begin if cont else 0
        annotations: List[Annotation] = self.get_entries(  # type: ignore
            a_type, cont, components)

        for annotation in annotations:
            # we provide span, text (and also tid) by default
            a_dict["span"].append((annotation.span.begin,
                                   annotation.span.end))
            a_dict["text"].append(annotation.text)

            for field in fields:
                if field in ("span", "text"):
                    continue
                if field == "context_span":
                    a_dict[field].append((annotation.span.begin - cont_begin,
                                          annotation.span.end - cont_begin))
                    continue

                a_dict[field].append(getattr(annotation, field))

            if unit is not None:
                while not self.index.in_span(data[unit]["tid"][unit_begin],
                                             annotation.span):
                    unit_begin += 1

                unit_span_begin = unit_begin
                unit_span_end = unit_span_begin + 1

                while self.index.in_span(data[unit]["tid"][unit_span_end],
                                         annotation.span):
                    unit_span_end += 1

                a_dict["unit_span"].append((unit_span_begin, unit_span_end))

        for key, value in a_dict.items():
            a_dict[key] = np.array(value)

        return a_dict

    def _generate_link_entry_data(
            self,
            a_type: Type[Link],
            a_args: Union[Dict, Iterable],
            data: Dict,
            cont: Optional[Annotation]) -> Dict:

        components, unit, fields = self._parse_request_args(a_type, a_args)

        if unit is not None:
            raise ValueError(f"Link entries cannot be indexed by {unit}.")

        a_dict: Dict[str, Any] = dict()
        for field in fields:
            a_dict[field] = []
        a_dict["parent"] = []
        a_dict["child"] = []

        links: List[Link] = self.get(a_type, cont, components)  # type: ignore

        for link in links:
            parent_type = link.ParentType.__name__
            child_type = link.ChildType.__name__

            if parent_type not in data.keys():
                raise KeyError(f"The Parent entry of {a_type} is not requested."
                               f" You should also request {parent_type} with "
                               f"{a_type}")
            if child_type not in data.keys():
                raise KeyError(f"The child entry of {a_type} is not requested."
                               f" You should also request {child_type} with "
                               f"{a_type}")

            a_dict["parent"].append(
                np.where(data[parent_type]["tid"] == link.parent)[0][0])
            a_dict["child"].append(
                np.where(data[child_type]["tid"] == link.child)[0][0])

            for field in fields:
                if field in ("parent", "child"):
                    continue

                a_dict[field].append(getattr(link, field))

        for key, value in a_dict.items():
            a_dict[key] = np.array(value)
        return a_dict

    def get_entries(self,
                    entry_type: Type[EntryType],
                    range_annotation: Optional[Annotation] = None,
                    components: Optional[Union[str, List[str]]] = None
                    ) -> Iterable[EntryType]:
        r"""Get ``entry_type`` entries from the span of ``range_annotation`` in
        a DataPack.

        Args:
            entry_type (type): The type of entries requested.
            range_annotation (Annotation, optional): The range of entries
                requested. If `None`, will return valid entries in the range of
                whole data_pack.
            components (str or list, optional): The component generating the
                entries requested. If `None`, will return valid entries
                generated by any component.
        """
        range_begin = range_annotation.span.begin if range_annotation else 0
        range_end = (range_annotation.span.end if range_annotation else
                     self.annotations[-1].span.end)

        # valid type
        valid_id = self.get_ids_by_type(entry_type)
        # valid component
        if components is not None:
            if isinstance(components, str):
                components = [components]
            valid_component_id: Set[int] = set()
            for component in components:
                valid_component_id |= self.get_ids_by_component(component)
            valid_id &= valid_component_id
        # valid span
        if range_annotation is not None:
            coverage_index = self.index.coverage_index(type(range_annotation),
                                                       entry_type)
            if coverage_index is not None:
                valid_id &= coverage_index[range_annotation.tid]

        if issubclass(entry_type, Annotation):
            begin_index = self.annotations.bisect(
                Annotation(self, range_begin, range_begin)
            )
            end_index = self.annotations.bisect(
                Annotation(self, range_end, range_end)
            )
            for annotation in self.annotations[begin_index: end_index]:
                if annotation.tid not in valid_id:
                    continue
                if (range_annotation is None or
                        self.index.in_span(annotation, range_annotation.span)):
                    yield annotation

        elif issubclass(entry_type, (Link, Group)):
            for entry_id in valid_id:
                entry: EntryType = self.get_entry(entry_id)  # type: ignore
                if (range_annotation is None or
                        self.index.in_span(entry, range_annotation.span)):
                    yield entry

        elif issubclass(entry_type, Query):
            for entry_id in valid_id:
                entry: EntryType = self.get_entry(entry_id)  # type: ignore
                if (range_annotation is None or
                        self.index.in_span(entry, range_annotation.span)):
                    yield entry

    def get(self,
            entry_type: Type[EntryType],
            range_annotation: Optional[Annotation] = None,
            component: Optional[str] = None) -> Iterable[EntryType]:
        r"""Get ``entry_type`` entries from the span of ``range_annotation`` in
        a DataPack generated by component ``component``.

        Example:

            .. code-block:: python

                for sentence in input_pack.get(Sentence):
                    token_entries = input_pack.get(entry_type=Token,
                                                   range_annotation=sentence,
                                                   component=token_component)
                    ...

            In the above code snippet, we get entries of type ``Token`` within
            each ``sentence`` which were generated by ``token_component``

        Args:
            entry_type (type): The type of entries requested.
            range_annotation (Annotation, optional): The range of entries
                requested. If `None`, will return valid entries in the range of
                whole data_pack.
            component (str or list, optional): The component generating the
                entries requested. If `None`, will return valid entries
                generated by any component.
        """
        return self.get_entries(entry_type, range_annotation, component)


class DataIndex(BaseIndex):
    r"""A set of indexes used in :class:`DataPack`:

    #. :attr:`entry_index`, the index from each tid to the corresponding entry
    #. :attr:`type_index`, the index from each type to the entries of
       that type
    #. :attr:`component_index`, the index from each component to the
       entries generated by that component
    #. :attr:`link_index`, the index from child
       (:attr:`link_index["child_index"]`)and parent
       (:attr:`link_index["parent_index"]`) nodes to links
    #. :attr:`group_index`, the index from group members to groups.
    #. :attr:`_coverage_index`, the index that maps from an annotation to
       the entries it covers. :attr:`_coverage_index` is a dict of dict, where
       the key is a tuple of the outer entry type and the inner entry type.
       The outer entry type should be an annotation type. The value is a dict,
       where the key is the tid of the outer entry, and the value is a set of
       tids that are covered by the outer entry.

    """

    def __init__(self):
        super().__init__()
        self._coverage_index: Dict[Tuple[Type[Annotation], Type[EntryType]],
                                   Dict[int, Set[int]]] = dict()
        self._coverage_index_valid = True

    @property
    def coverage_index_is_valid(self):
        return self._coverage_index_valid

    def activate_coverage_index(self):
        self._coverage_index_valid = True

    def deactivate_coverage_index(self):
        self._coverage_index_valid = False

    def coverage_index(
            self,
            outer_type: Type[Annotation],
            inner_type: Type[EntryType]) -> Optional[Dict[int, Set[int]]]:
        r"""Get the coverage index from ``outer_type`` to ``inner_type``.

        Args:
            outer_type (type): an annotation type.
            inner_type (type): an entry type.

        Returns:
            If the coverage index does not exist, return `None`. Otherwise,
            return a dict.
        """
        if not self.coverage_index_is_valid:
            return None
        return self._coverage_index.get((outer_type, inner_type))

    # To many call to data_pack?
    def build_coverage_index(
            self,
            data_pack: DataPack,
            outer_type: Type[Annotation],
            inner_type: Type[EntryType]):
        r"""Build the coverage index from ``outer_type`` to ``inner_type``.

        Args:
            data_pack (DataPack): The data pack to build coverage for.
            outer_type (type): an annotation type.
            inner_type (type): an entry type.
        """
        if not self.coverage_index_is_valid:
            self._coverage_index = dict()

        # prevent the index from being used during construction
        self.deactivate_coverage_index()

        self._coverage_index[(outer_type, inner_type)] = dict()
        for range_annotation in data_pack.get_entries_by_type(outer_type):
            if isinstance(range_annotation, Annotation):
                entries = data_pack.get_entries(inner_type, range_annotation)
                entry_ids = {e.tid for e in entries}
                self._coverage_index[
                    (outer_type, inner_type)][range_annotation.tid] = entry_ids

        self.activate_coverage_index()

    def have_overlap(self,
                     entry1: Union[Annotation, int],
                     entry2: Union[Annotation, int]) -> bool:
        r"""Check whether the two annotations have overlap in span.

        Args:
            entry1 (str or Annotation): An :class:`Annotation` object to be
                checked, or the tid of the Annotation.
            entry2 (str or Annotation): Another :class:`Annotation` object to be
                checked, or the tid of the Annotation.
        """
        entry1_: Annotation = self._entry_index[
            entry1] if isinstance(entry1, (int, np.integer)) else entry1
        entry2_: Annotation = self._entry_index[
            entry2] if isinstance(entry2, (int, np.integer)) else entry1

        if not isinstance(entry1_, Annotation):
            raise TypeError(f"'entry1' should be an instance of Annotation,"
                            f" but get {type(entry1)}")

        if not isinstance(entry2_, Annotation):
            raise TypeError(f"'entry2' should be an instance of Annotation,"
                            f" but get {type(entry2)}")

        return not (entry1_.span.begin >= entry2_.span.end or
                    entry1_.span.end <= entry2_.span.begin)

    def in_span(self,
                inner_entry: Union[int, Entry],
                span: Span) -> bool:
        r"""Check whether the ``inner entry`` is within the given ``span``. Link
        entries are considered in a span if both the parent and the child are
        within the span. Group entries are considered in a span if all the
        members are within the span.

        Args:
            inner_entry (int or Entry): The inner entry object to be checked
             whether it is within ``span``. The argument can be the entry id
             or the entry object itself.
            span (Span): A :class:`Span` object to be checked. We will check
                whether the ``inner_entry`` is within this span.
        """
        # The reason of this check is that the get_data method will use numpy
        # integers. This might create problems when other unexpected integers
        # are used.
        if isinstance(inner_entry, (int, np.integer)):
            inner_entry = self._entry_index[inner_entry]

        if isinstance(inner_entry, Annotation):
            inner_begin = inner_entry.span.begin
            inner_end = inner_entry.span.end
        elif isinstance(inner_entry, Link):
            child = inner_entry.get_child()
            parent = inner_entry.get_parent()

            if (not isinstance(child, Annotation)
                    or not isinstance(parent, Annotation)):
                # Cannot check in_span for non-annotations.
                return False

            child_: Annotation = child
            parent_: Annotation = parent

            inner_begin = min(child_.span.begin, parent_.span.begin)
            inner_end = max(child_.span.end, parent_.span.end)
        elif isinstance(inner_entry, Group):
            inner_begin = -1
            inner_end = -1
            for mem in inner_entry.get_members():
                if not isinstance(mem, Annotation):
                    # Cannot check in_span for non-annotations.
                    return False

                mem_: Annotation = mem
                if inner_begin == -1:
                    inner_begin = mem_.span.begin
                inner_begin = min(inner_begin, mem_.span.begin)
                inner_end = max(inner_end, mem_.span.end)
        else:
            raise ValueError(
                f"Invalid entry type {type(inner_entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )
        return inner_begin >= span.begin and inner_end <= span.end
