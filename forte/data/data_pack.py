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

import logging
from typing import (Dict, Iterable, Iterator, List, Optional, Type, Union, Any,
                    Set, Callable, Tuple)

import numpy as np
from sortedcontainers import SortedList

from forte.common.exception import ProcessExecutionException
from forte.data import data_utils_io
from forte.data.base_pack import BaseMeta, BasePack
from forte.data.index import BaseIndex
from forte.data.ontology.core import Entry
from forte.data.ontology.core import EntryType
from forte.data.ontology.top import (
    Annotation, Link, Group, SinglePackEntries, Generics)
from forte.data.span import Span
from forte.data.types import ReplaceOperationsType, DataRequest

logger = logging.getLogger(__name__)

__all__ = [
    "Meta",
    "DataPack",
]


class Meta(BaseMeta):
    r"""Basic Meta information associated with each instance of
    :class:`~forte.data.data_pack.DataPack`.

    Args:
        pack_name:  An name to identify the data pack, which is helpful in
           situation like serialization. It is suggested that the packs should
           have different doc ids.
        language: The language used by this data pack, default is English.
        span_unit: The unit used for interpreting the Span object of this
          data pack. Default is character.
    """

    def __init__(self, pack_name: Optional[str] = None,
                 language: str = 'eng', span_unit: str = 'character'):
        super().__init__(pack_name)
        self.language = language
        self.span_unit = span_unit


class DataPack(BasePack[Entry, Link, Group]):
    # pylint: disable=too-many-public-methods
    r"""A :class:`DataPack` contains a piece of natural language text and a
    collection of NLP entries (annotations, links, and groups). The natural
    language text could be a document, paragraph or in any other granularity.

    Args:
        pack_name (str, optional): A name for this data pack.
    """

    def __init__(self, pack_name: Optional[str] = None):
        super().__init__(pack_name)
        self._text = ""

        self.annotations: SortedList[Annotation] = SortedList()
        self.links: SortedList[Link] = SortedList()
        self.groups: SortedList[Group] = SortedList()
        self.generics: SortedList[Generics] = SortedList()

        self.__replace_back_operations: ReplaceOperationsType = []
        self.__processed_original_spans: List[Tuple[Span, Span]] = []
        self.__orig_text_len: int = 0

        self._index: DataIndex = DataIndex()

    def __getstate__(self):
        r"""
        In serialization,
            1) will serialize the annotation sorted list as a normal list;
            2) will not serialize the indices
        """
        state = super().__getstate__()
        state['annotations'] = list(state['annotations'])
        state['links'] = list(state['links'])
        state['groups'] = list(state['groups'])
        state['generics'] = list(state['generics'])
        return state

    def __setstate__(self, state):
        r"""
        In deserialization, we
            1) transform the annotation list back to a sorted list;
            2) initialize the indexes.
            3) Obtain the pack ids.
        """
        super().__setstate__(state)

        # For backward compatibility.
        if 'replace_back_operations' in self.__dict__:
            self.__replace_back_operations = self.__dict__.pop(
                'replace_back_operations')
        if 'processed_original_spans' in self.__dict__:
            self.__processed_original_spans = self.__dict__.pop(
                'processed_original_spans')
        if 'orig_text_len' in self.__dict__:
            self.__orig_text_len = self.__dict__.pop('orig_text_len')

        self.annotations = SortedList(self.annotations)
        self.links = SortedList(self.links)
        self.groups = SortedList(self.groups)
        self.generics = SortedList(self.generics)

        self._index = DataIndex()
        self._index.update_basic_index(list(self.annotations))
        self._index.update_basic_index(list(self.links))
        self._index.update_basic_index(list(self.groups))
        self._index.update_basic_index(list(self.generics))

        for a in self.annotations:
            a.set_pack(self)

        for a in self.links:
            a.set_pack(self)

        for a in self.groups:
            a.set_pack(self)

        for a in self.generics:
            a.set_pack(self)

    def __iter__(self):
        yield from self.annotations
        yield from self.links
        yield from self.groups
        yield from self.generics

    def _init_meta(self, pack_name: Optional[str] = None) -> Meta:
        return Meta(pack_name)

    def _validate(self, entry: EntryType) -> bool:
        return isinstance(entry, SinglePackEntries)

    @property
    def text(self) -> str:
        r"""Return the text of the data pack"""
        return self._text

    @property
    def all_annotations(self) -> Iterator[Annotation]:
        """
        An iterator of all annotations in this data pack.

        Returns: Iterator of all annotations, of
          type :class:"~forte.data.ontology.top.Annotation".

        """
        yield from self.annotations

    @property
    def num_annotations(self) -> int:
        """
        Number of annotations in this data pack.

        Returns: (int) Number of the links.

        """
        return len(self.annotations)

    @property
    def all_links(self) -> Iterator[Link]:
        """
        An iterator of all links in this data pack.

        Returns: Iterator of all links, of
          type :class:"~forte.data.ontology.top.Link".

        """
        yield from self.links

    @property
    def num_links(self) -> int:
        """
        Number of links in this data pack.

        Returns: Number of the links.

        """
        return len(self.links)

    @property
    def all_groups(self) -> Iterator[Group]:
        """
        An iterator of all groups in this data pack.

        Returns: Iterator of all groups, of
          type :class:"~forte.data.ontology.top.Group".

        """
        yield from self.groups

    @property
    def num_groups(self):
        """
        Number of groups in this data pack.

        Returns: Number of groups.

        """
        return len(self.groups)

    @property
    def all_generic_entries(self) -> Iterator[Generics]:
        """
        An iterator of all generic entries in this data pack.

        Returns: Iterator of generic

        """
        yield from self.generics

    @property
    def num_generics_entries(self):
        """
        Number of generics entries in this data pack.

        Returns: Number of generics entries.

        """
        return len(self.generics)

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

        if len(text) < len(self._text):
            raise ProcessExecutionException(
                "The new text is overwriting the original one with shorter "
                "length, which might cause unexpected behavior.")

        if len(self._text):
            logging.warning("Need to be cautious when changing the text of a "
                            "data pack, existing entries may get affected. ")

        span_ops = [] if replace_func is None else replace_func(text)

        # The spans should be mutually exclusive
        (
            self._text, self.__replace_back_operations,
            self.__processed_original_spans, self.__orig_text_len
        ) = data_utils_io.modify_text_and_track_ops(text, span_ops)

    def get_original_text(self):
        r"""Get original unmodified text from the :class:`DataPack` object.

        Returns:
            Original text after applying the `replace_back_operations` of
            :class:`DataPack` object to the modified text
        """
        original_text, _, _, _ = data_utils_io.modify_text_and_track_ops(
            self._text, self.__replace_back_operations)
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
            if len(self.__processed_original_spans) == 0:
                return input_index

            len_processed_text = len(self._text)
            orig_index = None
            prev_end = 0
            for (
                    inverse_span,
                    original_span) in self.__processed_original_spans:
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
                inverse_span, original_span = self.__processed_original_spans[
                    -1]
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

    @classmethod
    def deserialize(cls, data_pack_string: str) -> "DataPack":
        """
        Deserialize a Data Pack from a string. This internally calls the
        internal :meth:`~forte.data.base_pack.BasePack._deserialize` function
        from :class:`~forte.data.base_pack.BasePack`.

        Args:
            data_pack_string: The serialized string of a data pack to be
              deserialized.

        Returns:
            An data pack object deserialized from the string.
        """
        return cls._deserialize(data_pack_string)

    def _add_entry(self, entry: EntryType) -> EntryType:
        r"""Force add an :class:`~forte.data.ontology.core.Entry` object to the
        :class:`DataPack` object. Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.core.Entry`
                object to be added to the pack.

        Returns:
            The input entry itself
        """
        return self.__add_entry_with_check(entry, True)

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

            begin, end = entry.span.begin, entry.span.end

            if begin < 0:
                raise ValueError(f'The begin {begin} is smaller than 0, this'
                                 f'is not a valid begin.')

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
                    raise ValueError(
                        f"The end {end} of span is greater than the text "
                        f"length {len(self.text)}, which is invalid. The "
                        f"problematic entry is of type {entry.__class__} "
                        f"at [{begin}:{end}]"
                    )

        elif isinstance(entry, Link):
            target = self.links
        elif isinstance(entry, Group):
            target = self.groups
        elif isinstance(entry, Generics):
            target = self.generics
        else:
            raise ValueError(
                f"Invalid entry type {type(entry)}. A valid entry "
                f"should be an instance of Annotation, Link, Group of Generics."
            )

        # TODO: duplicate is ill-defined.
        add_new = allow_duplicate or (entry not in target)

        if add_new:
            target.add(entry)

            # update the data pack index if needed
            self._index.update_basic_index([entry])
            if self._index.link_index_on and isinstance(entry, Link):
                self._index.update_link_index([entry])
            if self._index.group_index_on and isinstance(entry, Group):
                self._index.update_group_index([entry])
            self._index.deactivate_coverage_index()

            self._pending_entries.pop(entry.tid)

            return entry
        else:
            return target[target.index(entry)]

    def delete_entry(self, entry: EntryType):
        r"""Delete an :class:`~forte.data.ontology.core.Entry` object from the
        :class:`DataPack`. This find out the entry in the index and remove it
        from the index. Note that entries will only appear in the index if
        `add_entry` (or _add_entry_with_check) is called.

        Please note that deleting a entry do not guarantee the deletion of
        the related entries.

        Args:
            entry (Entry): An :class:`~forte.data.ontology.core.Entry`
                object to be deleted from the pack.

        """
        if isinstance(entry, Annotation):
            target = self.annotations
        elif isinstance(entry, Link):
            target = self.links
        elif isinstance(entry, Group):
            target = self.groups
        elif isinstance(entry, Generics):
            target = self.generics
        else:
            raise ValueError(
                f"Invalid entry type {type(entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )

        begin: int = target.bisect_left(entry)

        index_to_remove = -1
        for i, e in enumerate(target[begin:]):
            if e.tid == entry.tid:
                index_to_remove = begin + i
                break

        if index_to_remove < 0:
            logger.warning(
                "The entry with id %d that you are trying to removed "
                "does not exists in the data pack's index. Probably it is "
                "created but not added in the first place.", entry.tid)
        else:
            target.pop(index_to_remove)

        # update basic index
        self._index.remove_entry(entry)

        # set other index invalid
        self._index.turn_link_index_switch(on=False)
        self._index.turn_group_index_switch(on=False)
        self._index.deactivate_coverage_index()

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
                data from the (`offset` + 1)th instance.

        Returns:
            A data generator, which generates one piece of data (a dict
            containing the required entries, fields, and context).
        """
        annotation_types: Dict[Type[Annotation], Union[Dict, List]] = dict()
        link_types: Dict[Type[Link], Union[Dict, List]] = dict()
        group_types: Dict[Type[Group], Union[Dict, List]] = dict()
        generics_types: Dict[Type[Generics], Union[Dict, List]] = dict()

        if request is not None:
            for key, value in request.items():
                if issubclass(key, Annotation):
                    annotation_types[key] = value
                elif issubclass(key, Link):
                    link_types[key] = value
                elif issubclass(key, Group):
                    group_types[key] = value
                elif issubclass(key, Generics):
                    generics_types[key] = value

        context_args = annotation_types.get(context_type)

        context_components, _, context_fields = self._parse_request_args(
            context_type, context_args)

        valid_context_ids: Set[int] = self.get_ids_by_type_subtype(context_type)
        if context_components:
            valid_component_id: Set[int] = set()
            for component in context_components:
                valid_component_id |= self.get_ids_by_creator(component)
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

            # TODO: Getting Group based on range is not done yet.
            if group_types:
                raise NotImplementedError(
                    "Querying groups based on ranges is "
                    "currently not supported.")

            if generics_types:
                raise NotImplementedError(
                    "Querying generic types based on ranges is "
                    "currently not supported.")

            yield data

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

        annotation: Annotation
        for annotation in self.get(a_type, cont, components):
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
                while not self._index.in_span(data[unit]["tid"][unit_begin],
                                              annotation.span):
                    unit_begin += 1

                unit_span_begin = unit_begin
                unit_span_end = unit_span_begin + 1

                while self._index.in_span(data[unit]["tid"][unit_span_end],
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

        link: Link
        for link in self.get(a_type, cont, components):
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

    def build_coverage_for(
            self, context_type: Type[Annotation], covered_type: Type[EntryType]
    ):
        """
        User can call this function to build coverage index for specific types.
          The index provide a in-memory mapping from entries of `context_type`
          to the entries "covered" by it.
          See :class:`forte.data.data_pack.DataIndex` for more details.

        Args:
            context_type: The context/covering type.
            covered_type: The entry to find under the context type.

        """
        if self._index.coverage_index(
                context_type, covered_type) is None:
            self._index.build_coverage_index(
                self, context_type, covered_type
            )

    def iter_in_range(
            self, entry_type: Type[EntryType], range_annotation: Annotation
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
        use_coverage = self._index.coverage_index_is_valid
        coverage_index: Optional[Dict[int, Set[int]]] = {}

        if use_coverage:
            coverage_index = self._index.coverage_index(
                type(range_annotation), entry_type)
            if coverage_index is None:
                use_coverage = False

        if use_coverage and coverage_index is not None:
            for tid in coverage_index[range_annotation.tid]:
                yield self.get_entry(tid)  # type: ignore
        else:
            if issubclass(entry_type, Annotation):
                range_begin = (
                    range_annotation.span.begin if range_annotation else 0)
                range_end = (range_annotation.span.end if range_annotation else
                             self.annotations[-1].span.end)

                if issubclass(entry_type, Annotation):
                    temp_begin = Annotation(self, range_begin, range_begin)
                    begin_index = self.annotations.bisect(temp_begin)

                    temp_end = Annotation(self, range_end, range_end)
                    end_index = self.annotations.bisect(temp_end)

                    # Make sure these temporary annotations are not part of the
                    # actual data.
                    temp_begin.regret_creation()
                    temp_end.regret_creation()
                    yield from self.annotations[begin_index: end_index]
            elif issubclass(entry_type, Link):
                for link in self.links:
                    if self._index.in_span(link, range_annotation.span):
                        yield link
            elif issubclass(entry_type, Group):
                for group in self.groups:
                    if self._index.in_span(group, range_annotation.span):
                        yield group

    def get(self, entry_type: Type[EntryType],  # type: ignore
            range_annotation: Optional[Annotation] = None,
            components: Optional[Union[str, Iterable[str]]] = None,
            include_sub_type=True
            ) -> Iterable[EntryType]:
        r"""This function is used to get data from a data pack with various
        methods.

        Depending on the provided arguments, the function will perform several
        different filtering of the returned data.

        The `entry_type` is mandatory, where all the entries matching this type
        will be returned. The sub-types of the provided entry type will be
        also returned if `include_sub_type` is set to True (which is the
        default behavior).

        The `range_annotation` controls the search area of the sub-types. An
        entry `E` will be returned if :meth:`in_span(E, range_annotation`
        returns True. If this function is called frequently with queries
        related to the `range_annotation`, please consider to build the coverage
        index regarding the related entry types.

        The `components` list will filter the results by the `component` (i.e
        the creator of the entry). If `components` is provided, only the entries
        created by one of the `components` will be returned.

        Example:

            .. code-block:: python

                # Iterate through all the sentences in the pack.
                for sentence in input_pack.get(Sentence):
                    # Take all tokens from a sentence created by NLTKTokenizer.
                    token_entries = input_pack.get(
                        entry_type=Token,
                        range_annotation=sentence,
                        component='NLTKTokenizer')
                    ...

            In the above code snippet, we get entries of type ``Token`` within
            each ``sentence`` which were generated by ``NLTKTokenizer``. You
            can consider build coverage index between `Token` and `Sentence`
            if this snippet is frequently used.

        Args:
            entry_type (type): The type of entries requested.
            range_annotation (Annotation, optional): The range of entries
                requested. If `None`, will return valid entries in the range of
                whole data_pack.
            components (str or list, optional): The component (creator)
                generating the entries requested. If `None`, will return valid
                entries generated by any component.
            include_sub_type (bool): whether to consider the sub types of
                the provided entry type. Default `True`.
        """

        def require_annotations() -> bool:
            if issubclass(entry_type, Annotation):
                return True
            if issubclass(entry_type, Link):
                return (issubclass(entry_type.ParentType, Annotation)
                        and issubclass(entry_type.ChildType, Annotation))
            if issubclass(entry_type, Group):
                return issubclass(entry_type.MemberType, Annotation)
            return False

        # If we don't have any annotations but the items to check requires them,
        # then we simply yield from an empty list.
        if len(self.annotations) == 0 and range_annotation is not None:
            if require_annotations():
                yield from []
                return

        # Valid entry ids based on type.
        all_types: Set[Type]
        if include_sub_type:
            all_types = self._expand_to_sub_types(entry_type)
        else:
            all_types = {entry_type}

        entry_iter: Iterator[Entry]
        if issubclass(entry_type, Generics):
            entry_iter = self.generics
        elif range_annotation is not None:
            if (issubclass(entry_type, Annotation)
                    or issubclass(entry_type, Link)
                    or issubclass(entry_type, Group)):
                entry_iter = self.iter_in_range(entry_type, range_annotation)
        elif issubclass(entry_type, Annotation):
            entry_iter = self.annotations
        elif issubclass(entry_type, Link):
            entry_iter = self.links
        elif issubclass(entry_type, Group):
            entry_iter = self.groups
        else:
            raise ValueError(
                f"The requested type {str(entry_type)} is not supported.")

        for entry in entry_iter:
            # Filter by type and components.
            if type(entry) not in all_types:
                continue
            if components is not None:
                if not self.is_created_by(entry, components):
                    continue
            yield entry  # type: ignore


class DataIndex(BaseIndex):
    r"""A set of indexes used in :class:`DataPack`, note that this class is
    used by the `DataPack` internally.

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
       `tid` that are covered by the outer entry. We say an Annotation A covers
       an entry E if one of the following condition is met:
       1. E is of Annotation type, and that E.begin >= A.begin, E.end <= E.end
       2. E is of Link type, and both E's parent and child node are Annotation
       that are covered by A.

    """

    def __init__(self):
        super().__init__()
        self._coverage_index: Dict[Tuple[Type[Annotation], Type[EntryType]],
                                   Dict[int, Set[int]]] = dict()
        self._coverage_index_valid = True

    def remove_entry(self, entry: EntryType):
        super().remove_entry(entry)
        self.deactivate_coverage_index()

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

    def build_coverage_index(
            self,
            data_pack: DataPack,
            outer_type: Type[Annotation],
            inner_type: Type[EntryType]):
        r"""Build the coverage index from ``outer_type`` to ``inner_type``.

        Args:
            data_pack (DataPack): The data pack to build coverage for.
            outer_type (type): an annotation type.
            inner_type (type): an entry type, can be Annotation, Link, Group.
        """
        if not isinstance(inner_type, (Annotation, Link, Group)):
            raise ValueError(f"Do not support coverage index for {inner_type}.")

        if not self.coverage_index_is_valid:
            self._coverage_index = dict()

        # prevent the index from being used during construction
        self.deactivate_coverage_index()

        # TODO: tests and documentations for the edge cases are missing. i.e. we
        #  are not clear about what would happen if the covered annotation
        #  is the same as the covering annotation, or if their spans are the
        #  same.
        self._coverage_index[(outer_type, inner_type)] = dict()
        for range_annotation in data_pack.get_entries_of(
                outer_type):
            if isinstance(range_annotation, Annotation):
                entries = data_pack.get(inner_type, range_annotation)
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

    def in_span(self, inner_entry: Union[int, Entry], span: Span) -> bool:
        r"""Check whether the ``inner entry`` is within the given ``span``. The
        criterion are as followed:

        Annotation entries: they are considered in a span if the begin is not
        smaller than `span.begin` and the end is not larger than `span.end`.

        Link entries: if the parent and child of the links are both
        `Annotation` type, this link will be considered in span if both parent
        and child are `in_span` of the provided `span`. If either the parent and
        the child is not of type `Annotation`, this function will always return
        `False`.

        Group entries: if the child type of the group is `Annotation` type,
        then the group will be considered in span if all the elements are
        `in_span` of the provided `span`. If the child type is not `Annotation`
        type, this function will always return `False`.

        Other entries (i.e Generics): they will not be considered `in_span` of
        any spans. The function will always return `False`.

        Args:
            inner_entry (int or Entry): The inner entry object to be checked
             whether it is within ``span``. The argument can be the entry id
             or the entry object itself.
            span (Span): A :class:`Span` object to be checked. We will check
                whether the ``inner_entry`` is within this span.

        Returns:
            True if the `inner_entry` is considered to be in span of the
            provided span.
        """
        # The reason of this check is that the get_data method will use numpy
        # integers. This might create problems when other unexpected integers
        # are used.
        if isinstance(inner_entry, (int, np.integer)):
            inner_entry = self._entry_index[inner_entry]

        inner_begin = -1
        inner_end = -1

        if isinstance(inner_entry, Annotation):
            inner_begin = inner_entry.span.begin
            inner_end = inner_entry.span.end
        elif isinstance(inner_entry, Link):
            if not issubclass(inner_entry.ParentType, Annotation):
                return False

            if not issubclass(inner_entry.ChildType, Annotation):
                return False

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
            if not issubclass(inner_entry.MemberType, Annotation):
                return False

            for mem in inner_entry.get_members():
                mem_: Annotation = mem  # type: ignore
                if inner_begin == -1:
                    inner_begin = mem_.span.begin
                inner_begin = min(inner_begin, mem_.span.begin)
                inner_end = max(inner_end, mem_.span.end)
        else:
            # Generics or other user defined types will not be check here.
            return False
        return inner_begin >= span.begin and inner_end <= span.end
