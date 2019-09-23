import copy
import logging
from typing import (
    Dict, Iterable, Iterator, List, Tuple, Optional, Type, Union,
    Any, Set, Callable, NewType)

import numpy as np
from sortedcontainers import SortedList

from forte.data.base_pack import BaseIndex, BaseMeta, BasePack
from forte.data.ontology import (
    Entry, EntryType, Annotation, Link, Group, Span)

logger = logging.getLogger(__name__)

__all__ = [
    "Meta",
    "DataIndex",
    "DataPack",
    "ReplaceOperationsType",
    "DataRequest",
]

ReplaceOperationsType = List[Tuple[Tuple[int, int], str]]
DataRequest = Dict[Type[Entry], Union[Dict, List]]


class Meta(BaseMeta):
    """
    Meta information of a datapack.
    """

    def __init__(
            self,
            doc_id: Optional[str] = None,
            language: str = 'eng',
            span_unit: str = 'character'
    ):
        super().__init__(doc_id)
        self.language = language
        self.span_unit = span_unit


class DataPack(BasePack):
    """
    A :class:`DataPack' contains a piece of natural language text and a
    collection of NLP entries (annotations, links, and groups). The natural
    language text could be a document, paragraph or in any other granularity.

    Args:
        doc_id (str, optional): A universal id of this data pack.
    """

    def __init__(self, doc_id: Optional[str] = None):
        super().__init__()
        self._text = ""

        self.annotations: SortedList[Annotation] = SortedList()
        self.links: List[Link] = []
        self.groups: List[Group] = []

        self.inverse_replace_operations: ReplaceOperationsType = []

        self.index: DataIndex = DataIndex(self)
        self.meta: Meta = Meta(doc_id)

    def __getstate__(self):
        """
        In serialization, 1) will serialize the annotation sorted list as a
        normal list; 2) will not serialize the indexes
        """
        state = self.__dict__.copy()
        state['annotations'] = list(state['annotations'])
        state.pop('index')
        return state

    def __setstate__(self, state):
        """
        In deserialization, we 1) transform the annotation list back to a
        sorted list; 2) initialize the indexes.
        """
        self.__dict__.update(state)
        self.annotations = SortedList(self.annotations)
        self.index = DataIndex(self)
        self.index.update_basic_index(list(self.annotations))
        self.index.update_basic_index(self.links)
        self.index.update_basic_index(self.groups)

    @property
    def text(self):
        return self._text

    def set_text(self,
                 text: str,
                 replace_func: Optional[
                     Callable[[str], ReplaceOperationsType]] = None
                 ):

        if len(self._text) > 0:
            logger.warning("The new text is overwriting the original one, "
                           "which might cause unexpected behavior.")

        span_ops = [] if replace_func is None else replace_func(text)

        # Sorting the spans such that the order of replacement strings
        # is maintained - utilizing the stable sort property of python sort
        span_ops.sort(key=lambda item: item[0])

        span_ops = [(Span(op[0], op[1]), replacement)
                    for op, replacement in span_ops]

        # The spans should be mutually exclusive
        inverse_operations = []
        increment = 0
        prev_span_end = 0
        mod_text = text
        for span, replacement in span_ops:
            if span.begin < 0 or span.end < 0:
                raise ValueError(
                    "Negative indexing not supported")
            if span.begin > len(text) or span.end > len(text):
                raise ValueError(
                    "One of the span indices are outside the string length")
            if span.end < span.begin:
                print(span.begin, span.end)
                raise ValueError(
                    "One of the end indices is lesser than start index")
            if span.begin < prev_span_end:
                raise ValueError(
                    "The replacement spans should be mutually exclusive")
            span_begin = span.begin + increment
            span_end = span.end + increment
            original_span_text = mod_text[span_begin: span_end]
            mod_text = mod_text[:span_begin] + replacement + mod_text[span_end:]
            increment += len(replacement) - (span.end - span.begin)
            replacement_span = (span_begin, span_begin + len(replacement))
            inverse_operations.append((replacement_span, original_span_text))
            prev_span_end = span.end

        self._text = mod_text
        self.inverse_replace_operations = inverse_operations

    def add_or_get_entry(self, entry: EntryType) -> EntryType:
        """
        Try to add an :class:`Entry` object to the :class:`DataPack` object.
        If a same entry already exists, will return the existing entry
        instead of adding the new one. Note that we regard two entries to be
        same if their :meth:`eq` have the same return value, and users could
        override :meth:`eq` in their custom entry classes.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the datapack.

        Returns:
            If a same entry already exists, returns the existing
            entry. Otherwise, return the (input) entry just added.
        """
        return self.__add_entry(entry, False)

    def add_entry(self, entry: EntryType) -> EntryType:
        """
        Force add an :class:`Entry` object to the :class:`DataPack` object.
        Allow duplicate entries in a datapack.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the datapack.

        Returns:
            The input entry itself
        """
        return self.__add_entry(entry, True)

    def __add_entry(self, entry: EntryType,
                    allow_duplicate: bool = True) -> EntryType:
        """
        Internal method to add an :class:`Entry` object to the
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
            raise ValueError(
                f"Invalid entry type {type(entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )

        add_new = allow_duplicate or (entry not in target)

        if add_new:
            # add the entry to the target entry list
            entry.set_tid(str(self.internal_metas[entry.__class__].id_counter))
            entry.__set_component(self.__owner_component)

            if isinstance(target, list):
                target.append(entry)
            else:
                target.add(entry)
            self.internal_metas[entry.__class__].id_counter += 1

            # update the data pack index if needed
            self.index.update_basic_index([entry])
            if self.index.link_index_switch and isinstance(entry, Link):
                self.index.update_link_index([entry])
            if self.index.group_index_switch and isinstance(entry, Group):
                self.index.update_group_index([entry])
            self.index.deactivate_coverage_index()
            return entry
        else:
            return target[target.index(entry)]

    def delete_entry(self, entry: EntryType):
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
        self.index.entry_index.pop(entry.tid)
        self.index.type_index[type(entry)].remove(entry.tid)
        self.index.component_index[entry.__component].remove(entry.tid)
        # set other index invalid
        self.index.turn_link_index_switch(on=False)
        self.index.turn_group_index_switch(on=False)
        self.index.deactivate_coverage_index()

    def record_fields(self, fields: List[str], entry_type: Type[Entry],
                      component: str):
        """Record in the internal meta that the ``entry_type`` generated by
        ``component`` have ``fields``.

        If ``component`` is "_ALL_", we will record ``fields`` for all existing
        components in the internal meta of ``entry_type``.
        # TODO: add explanation about component in documentation
        """
        fields.append("tid")
        if issubclass(entry_type, Annotation):
            fields.append("span")
        internal_meta = self.internal_metas[entry_type]

        if component == "_ALL_":
            for field_set in internal_meta.fields_created.values():
                field_set.update(fields)
        else:
            if internal_meta.default_component is None:
                internal_meta.default_component = component
            internal_meta.fields_created[component].update(fields)

    def get_data(
            self,
            context_type: Type[Annotation],
            request: Optional[DataRequest] = None,
            skip_k: int = 0
    ) -> Iterator[Dict[str, Any]]:
        """
        Example:

            .. code-block:: python

                requests = {
                    base_ontology.Sentence:
                        {
                            "component": ["dummy"],
                            "fields": ["speaker"],
                        },
                    base_ontology.Token: ["pos_tag", "sense""],
                    base_ontology.EntityMention: {
                        "unit": "Token",
                    },
                }
                pack.get_data("sentence", requests)

        Args:
            context_type (str): The granularity of the data context, which
                could be any Annotation type.
            request (dict): The entry types and fields required.
                The keys of the dict are the required entry types and the
                value should be either a list of field names or a dict.
                If the value is a dict, accepted items includes "fields",
                "component", and "unit". By setting "component" (a list), users
                can specify the components by which the entries are generated.
                If "component" is not specified, will return entries generated
                by all components. By setting "unit" (a string), users can
                specify a unit by which the annotations are indexed.
                Note that for all annotations, "text" and "span" fields are
                given by default; for all links, "child" and "parent"
                fields are given by default.
            skip_k (int): Will skip the first k instances and generate
                data from the k + 1 instance.
        Returns:
            A data generator, which generates one piece of data (a dict
            containing the required annotations and context).
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
            context_type, context_args
        )

        valid_context_ids = self.get_ids_by_type(context_type)
        if context_components:
            valid_component_id: Set[str] = set()
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

            data = dict()
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
                            a_type, a_args, data, context
                        )
            if link_types:
                for l_type, l_args in link_types.items():
                    if l_type.__name__ in data.keys():
                        raise KeyError(
                            f"Requesting two types of entries with the "
                            f"same class name {l_type.__name__} at the "
                            f"same time is not allowed")
                    data[l_type.__name__] = self._generate_link_entry_data(
                        l_type, l_args, data, context
                    )

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

        # check the existence of fields
        for meta_key, meta_val in self.internal_metas.items():
            if issubclass(meta_key, a_type):
                for meta_c, meta_f in meta_val.fields_created.items():
                    if components is None or meta_c in components:
                        if not fields.issubset(meta_f):
                            raise KeyError(
                                f"The {a_type} generated by {meta_c} doesn't "
                                f"have the fields requested.")

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
        annotations = self.get_entries(a_type, cont, components)

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

        links = self.get(a_type, cont, components)

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
        """
        Get ``entry_type`` entries from the span of ``range_annotation`` in a
        DataPack.

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
            valid_component_id: Set[str] = set()
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
                Annotation(self, range_begin, -1)
            )
            end_index = self.annotations.bisect(
                Annotation(self, range_end, -1)
            )
            for annotation in self.annotations[begin_index: end_index]:
                if annotation.tid not in valid_id:
                    continue
                if (range_annotation is None or
                        self.index.in_span(annotation, range_annotation.span)):
                    yield annotation

        elif issubclass(entry_type, (Link, Group)):
            for entry_id in valid_id:
                entry = self.get_entry_by_id(entry_id)
                if (range_annotation is None or
                        self.index.in_span(entry, range_annotation.span)):
                    yield entry

    def get_entry_by_id(self, tid: str):
        """
        Look up the entry_index with key ``tid``.
        """
        entry = self.index.entry_index.get(tid)
        if entry is None:
            raise KeyError(
                f"There is no entry with tid '{tid}'' in this datapack")
        return entry

    def get_ids_by_component(self, component: str) -> Set[str]:
        """
        Look up the component_index with key ``component``.
        """
        entry_set = self.index.component_index[component]
        if len(entry_set) == 0:
            logging.warning("There is no entry generated by '%s' "
                            "in this datapack", component)
        return entry_set

    def get_entries_by_component(self, component: str) -> Set[Entry]:
        return {self.get_entry_by_id(tid)
                for tid in self.get_ids_by_component(component)}

    def get_ids_by_type(self, tp: Type[EntryType]) -> Set[str]:
        """
        Look up the type_index with key ``tp``.

        Returns:
             A set of entry tids. The entries are instances of tp (and also
             includes instances of the subclasses of tp).
        """
        subclass_index = set()
        for index_key, index_val in self.index.type_index.items():
            if issubclass(index_key, tp):
                subclass_index.update(index_val)

        if len(subclass_index) == 0:
            logging.warning("There is no %s type entry in this datapack", tp)
        return subclass_index

    def get_entries_by_type(self, tp: Type[EntryType]) -> Set[EntryType]:
        entries: Set = set()
        for tid in self.get_ids_by_type(tp):
            entry = self.get_entry_by_id(tid)
            if isinstance(entry, tp):
                entries.add(entry)
        return entries

    def get_links_by_parent(self, parent: Union[str, Entry]) -> Set[Link]:
        links = set()
        if isinstance(parent, Entry):
            tid = parent.tid
            if tid is None:
                raise ValueError(f"Argument parent has no tid. "
                                 f"Have you add this entry into the datapack?")
        else:
            tid = parent
        for tid in self.index.link_index(tid, as_parent=True):
            entry = self.get_entry_by_id(tid)
            if isinstance(entry, Link):
                links.add(entry)
        return links

    def get_links_by_child(self, child: Union[str, Entry]) -> Set[Link]:
        links = set()
        if isinstance(child, Entry):
            tid = child.tid
            if tid is None:
                raise ValueError(f"Argument child has no tid. "
                                 f"Have you add this entry into the datapack?")
        else:
            tid = child
        for tid in self.index.link_index(tid, as_parent=False):
            entry = self.get_entry_by_id(tid)
            if isinstance(entry, Link):
                links.add(entry)
        return links

    def get_groups_by_member(self, member: Union[str, Entry]) -> Set[Group]:
        groups = set()
        if isinstance(member, Entry):
            tid = member.tid
            if tid is None:
                raise ValueError(f"Argument member has no tid. "
                                 f"Have you add this entry into the datapack?")
        else:
            tid = member
        for tid in self.index.group_index(tid):
            entry = self.get_entry_by_id(tid)
            if isinstance(entry, Group):
                groups.add(entry)
        return groups

    def get(self,
            entry_type: Type[EntryType],
            range_annotation: Optional[Annotation] = None,
            component: Optional[str] = None) -> Iterable[EntryType]:
        return self.get_entries(entry_type, range_annotation, component)

    def view(self):
        return copy.deepcopy(self)


class DataIndex(BaseIndex[DataPack]):
    def __init__(self, data_pack):
        super().__init__(data_pack)
        self._coverage_index: Dict[Tuple[Type[Annotation], Type[EntryType]],
                                   Dict[str, Set[str]]] = dict()
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
            outter_type: Type[Annotation],
            inner_type: Type[EntryType]) -> Optional[Dict[str, Set[str]]]:
        if not self.coverage_index_is_valid:
            return None
        return self._coverage_index.get((outter_type, inner_type))

    def build_coverage_index(self,
                             outter_type: Type[Annotation],
                             inner_type: Type[EntryType]):
        if not self.coverage_index_is_valid:
            self._coverage_index = dict()

        # prevent the index from being used during construction
        self.deactivate_coverage_index()

        self._coverage_index[(outter_type, inner_type)] = dict()
        for range_annotation in self.data_pack.get_entries_by_type(outter_type):
            entries = self.data_pack.get_entries(inner_type, range_annotation)
            entry_ids = {e.tid for e in entries}
            self._coverage_index[(outter_type,
                                  inner_type)][range_annotation.tid] = entry_ids

        self.activate_coverage_index()

    def have_overlap(self,
                     entry1: Union[Annotation, str],
                     entry2: Union[Annotation, str]) -> bool:
        """Check whether the two annotations have overlap in span.

        Args:
            entry1 (str or Annotation): An :class:`Annotation` object to be
                checked, or the tid of the Annotation.
            entry2 (str or Annotation): Another :class:`Annotation` object to be
                checked, or the tid of the Annotation.
        """
        if isinstance(entry1, str):
            e = self.entry_index[entry1]
            if not isinstance(e, Annotation):
                raise TypeError(f"'entry1' should be an instance of Annotation,"
                                f" but get {type(e)}")
            entry1 = e

        if not isinstance(entry1, Annotation):
            raise TypeError(f"'entry1' should be an instance of Annotation,"
                            f" but get {type(entry1)}")

        if isinstance(entry2, str):
            e = self.entry_index[entry2]
            if not isinstance(e, Annotation):
                raise TypeError(f"'entry2' should be an instance of Annotation,"
                                f" but get {type(e)}")
            entry2 = e

        if not isinstance(entry2, Annotation):
            raise TypeError(f"'entry2' should be an instance of Annotation,"
                            f" but get {type(entry2)}")

        return not (entry1.span.begin >= entry2.span.end or
                    entry1.span.end <= entry2.span.begin)

    def in_span(self,
                inner_entry: Union[str, Entry],
                span: Span) -> bool:
        """Check whether the ``inner entry`` is within the given ``span``.
        Link entries are considered in a span if both the
        parent and the child are within the span. Group entries are
        considered in a span if all the members are within the span.

        Args:
            inner_entry (str or Entry): An :class:`Entry` object to be checked.
                We will check whether this entry is within ``span``.
            span (Span): A :class:`Span` object to be checked. We will check
                whether the ``inner_entry`` is within this span.
        """

        if isinstance(inner_entry, str):
            inner_entry = self.entry_index[inner_entry]

        if isinstance(inner_entry, Annotation):
            inner_begin = inner_entry.span.begin
            inner_end = inner_entry.span.end
        elif isinstance(inner_entry, Link):
            child = inner_entry.get_child()
            parent = inner_entry.get_parent()
            inner_begin = min(child.span.begin, parent.span.begin)
            inner_end = max(child.span.end, parent.span.end)
        elif isinstance(inner_entry, Group):
            inner_begin = -1
            inner_end = -1
            for mem in inner_entry.get_members():
                if inner_begin == -1:
                    inner_begin = mem.span.begin
                inner_begin = min(inner_begin, mem.span.begin)
                inner_end = max(inner_end, mem.span.end)
        else:
            raise ValueError(
                f"Invalid entry type {type(inner_entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )
        return inner_begin >= span.begin and inner_end <= span.end
