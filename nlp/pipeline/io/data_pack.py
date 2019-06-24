""" This class defines the core interchange format, deals with basic operations
such as reading, writing, checking and indexing.
"""
import logging
import itertools
from collections import defaultdict
from typing import Union, Dict, Optional, List, Iterable, DefaultDict
import numpy as np
from sortedcontainers import SortedList

from nlp.pipeline.io.base_ontology import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class InternalMeta:
    def __init__(self):
        self.id_counter = 0
        self.fields_created = dict()
        self.default_component = None


class DataIndex:
    def __init__(self):
        self.entry_index = defaultdict(Entry)
        self.type_index = defaultdict(set)
        self.sentence_index = defaultdict(set)
        self.component_index = defaultdict(set)
        self.coverage_index: Dict[str, DefaultDict[str, set]] = dict()


class DataPack:
    """
    A :class:`DataPack' contains a piece of natural language text and a
    collection of NLP entries (annotations, links, and groups). The natural
    language text could be a document, paragraph or in any other granularity.

    Args:
        text (str, optional): A piece of natural language text.
        doc_id (str, optional): A universal id of this data pack.
    """

    def __init__(self, text: str = None, doc_id: str = None):
        self.annotations = SortedList()
        self.links = []
        self.groups = []
        self.meta = Meta(doc_id)
        self.text = text

        self.index = DataIndex()
        self.internal_metas = defaultdict(InternalMeta)

    def add_entry(self, entry: Entry, indexing: bool = True):
        """
        Try to add an :class:`Entry` object to the :class:`DataPack` object.
        If a same entry already exists, will not add the new one.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the datapack.
            indexing (bool): Whether to update the data pack index. Indexing is
                always suggested unless you are sure that your pipeline will
                never refer it.

        Returns:
            If a same annotation already exists, returns the tid of the
            existing annotation. Otherwise, return the tid of the annotation
            just added.
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

        if entry not in target:
            # add the entry to the target entry list
            name = entry.__class__.__name__
            if entry.tid is None:
                entry.set_tid(str(self.internal_metas[name].id_counter))
            if isinstance(target, list):
                target.append(entry)
            else:
                target.add(entry)
            self.internal_metas[name].id_counter += 1

            # update the data pack index if needed
            if indexing:
                self.index.entry_index[entry.tid] = entry
                self.index.type_index[name].add(entry.tid)
                self.index.component_index[entry.component].add(entry.tid)

            return entry.tid
        # logger.debug(f"Annotation already exist {annotation.tid}")
        return target[target.index(entry)].tid

    def record_fields(self, fields: list, component: str, entry_type: str):
        """Record in the internal meta that ``component`` has generated
        ``fields`` for ``entry_type``.
        """
        if entry_type not in self.internal_metas.keys():
            self.internal_metas[entry_type].default_component = component

        # ensure to record entry_type if fields list is empty
        if component not in self.internal_metas[
            entry_type].fields_created.keys():
            self.internal_metas[entry_type].fields_created[component] = set()

        for field in fields:
            self.internal_metas[entry_type].fields_created[component].add(field)

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self.meta, k, v)

    def index_entries(self):
        """Build the data pack index. """
        for entry in itertools.chain(self.annotations, self.links, self.groups):
            name = entry.__class__.__name__
            self.index.entry_index[entry.tid] = entry
            self.index.type_index[name].add(entry.tid)
            self.index.component_index[entry.component].add(entry.tid)

        # index the entries in the span of sentence
        for i in range(len(self.annotations)):
            if self.annotations[i].tid in self.index.type_index["Sentence"]:
                for k in range(i, -1, -1):
                    if self._in_span(self.annotations[k],
                                     self.annotations[i].span):
                        self.index.sentence_index[
                            self.annotations[i].tid
                        ].add(self.annotations[k].tid)
                    else:
                        break
                for k in range(i, len(self.annotations)):
                    if self._in_span(self.annotations[k],
                                     self.annotations[i].span):
                        self.index.sentence_index[
                            self.annotations[i].tid
                        ].add(self.annotations[k].tid)
                    else:
                        break

        for entry in itertools.chain(self.links, self.groups):
            for sent_id in self.index.type_index["Sentence"]:
                sent = self.index.entry_index[sent_id]
                if hasattr(sent, "span") and self._in_span(entry, sent.span):
                    self.index.sentence_index[sent_id].add(entry.tid)

    def index_annotation_coverage(self,
                                     outter_type: Optional[str] = None,
                                     inner_type: Optional[str] = None):
        """
        Index the coverage relationship from annotations of outter_type to
        annotations of inner_type, and store in
        ``self.index.coverage_index["outter_type-to-inner_type"]``. An outter
        annotation is considered to cover an inner annotation if inner.begin
        >= outter.begin and inner.end <= outter.end.

        Args:
            outter_type (str, optional): The type of the outter annotations. If
                `None`, the outter annotations could be of all types, and the
                index name will be "all-to-inner_type".
            inner_type (str, optional): The type of the inner annotations. If
                `None`, the inner annotations could be of all types, and the
                index name will be "outter_type-to-all".
        """

        dict_name = outter_type if outter_type else "all"
        outter_ids = self.index.type_index[outter_type] if outter_type else None
        dict_name += "-to-"
        dict_name += inner_type if inner_type else "all"
        inner_ids = self.index.type_index[inner_type] if inner_type else None
        self.index.coverage_index[dict_name] = defaultdict(set)

        for i in range(len(self.annotations)):
            if outter_ids is None or self.annotations[i].tid in outter_ids:
                for k in range(i, -1, -1):
                    if inner_ids is None or self.annotations[k].tid in inner_ids:
                        if self._in_span(self.annotations[k],
                                         self.annotations[i].span):
                            self.index.coverage_index[dict_name][
                                self.annotations[i].tid
                            ].add(self.annotations[k].tid)
                        else:
                            break
                for k in range(i, len(self.annotations)):
                    if inner_ids is None or self.annotations[k].tid in inner_ids:
                        if self._in_span(self.annotations[k],
                                         self.annotations[i].span):
                            self.index.coverage_index[dict_name][
                                self.annotations[i].tid
                            ].add(self.annotations[k].tid)
                        else:
                            break

    def _in_span(self,
                 inner_entry: Union[str, Entry],
                 span: Span) -> bool:
        """Check whether the ``inner annotation`` is within the span of the
        ``outer_entry``. Link entries are considered in a span if both the
        parent and the child are within the outer span. Group entries are
        considered in a span if all the members are within the outer span.

        Args:
            inner_entry (str or Entry): An :class:`Entry` object to be checked.
                We will check whether this entry is within the span of
                ``outer_entry``.
            outer_entry (str or Annotation): An :class:`Annotation` object
                to be checked. We will check whether the ``inner_entry`` is
                within the span of this entry.
        """

        if isinstance(inner_entry, str):
            inner_entry = self.index.entry_index.get(inner_entry)

        if isinstance(inner_entry, Annotation):
            inner_begin = inner_entry.span.begin
            inner_end = inner_entry.span.end
        elif isinstance(inner_entry, Link):
            child = self.index.entry_index.get(inner_entry.child)
            parent = self.index.entry_index.get(inner_entry.parent)
            inner_begin = min(child.span.begin, parent.span.begin)
            inner_end = min(child.span.end, parent.span.end)
        elif isinstance(inner_entry, Group):
            inner_begin = -1
            inner_end = -1
            for m_id in inner_entry.members:
                mem = self.index.entry_index.get(m_id)
                if inner_begin == -1:
                    inner_begin = mem.span.begin
                inner_begin = min(inner_begin, mem.span.begin)
                inner_end = min(inner_end, mem.span.end)
        else:
            raise ValueError(
                f"Invalid entry type {type(inner_entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )
        return inner_begin >= span.begin and inner_end <= span.end

    def get_data(
            self,
            context_type: str,
            annotation_types: Dict[str, Union[Dict, Iterable]] = None,
            link_types: Dict[str, Union[Dict, Iterable]] = None,
            group_types: Dict[str, Union[Dict, Iterable]] = None,
            offset: int = 0
    ) -> Iterable[Dict]:
        """

        Args:
            context_type (str): The granularity of the data context, which
                could be either `"sentence"` or `"document"`
            annotation_types (dict): The annotation types and fields required.
                The keys of the dict are the required annotation types and the
                values could be a list, set, or tuple of field names. Users can
                also specify the component from which the annotations are
                generated.
            link_types (dict): The link types and fields required.
                The keys of the dict are the required link types and the
                values could be a list, set, or tuple of field names. Users can
                also specify the component from which the annotations are
                generated.
            group_types (dict): The group types and fields required.
                The keys of the dict are the required group types and the
                values could be a list, set, or tuple of field names. Users can
                also specify the component from which the annotations are
                generated.
            offset (int): Will skip the first `offset` instances and generate
                data from the `offset` + 1 instance.
        Returns:
            A data generator, which generates one piece of data (a dict
            containing the required annotations and context).
        """

        if context_type == "document":
            data = dict()
            data["context"] = self.text

            if annotation_types:
                for a_type, a_args in annotation_types.items():
                    data[a_type] = self._generate_annotation_entry_data(
                        a_type, a_args, None
                    )

            if link_types:
                for a_type, a_args in link_types.items():
                    data[a_type] = self._generate_link_entry_data(
                        a_type, a_args, None
                    )
            yield data

        elif context_type == "sentence":

            sent_meta = self.internal_metas.get("Sentence")
            if sent_meta is None:
                raise AttributeError(
                    f"Document '{self.meta.doc_id}' has no sentence "
                    f"annotations'"
                )

            sent_args = annotation_types.get("Sentence")

            sent_component, sent_fields = self._process_request_args(
                "Sentence", sent_args
            )

            valid_sent_ids = (self.index.type_index["Sentence"]
                              & self.index.component_index[sent_component])

            skipped = 0
            for sent in self.annotations:  # to maintain the order
                if sent.tid not in valid_sent_ids:
                    continue
                if skipped < offset:
                    skipped += 1
                    continue

                data = dict()
                data["context"] = self.text[sent.span.begin: sent.span.end]

                for field in sent_fields:
                    if field not in sent_meta.fields_created[sent_component]:
                        raise AttributeError(
                            f"Sentence annotation generated by "
                            f"'{sent_component}' has no field named '{field}'."
                        )

                    data[field] = getattr(sent, field)

                if annotation_types is not None:
                    for a_type, a_args in annotation_types.items():
                        if a_type == "Sentence":
                            continue

                        data[a_type] = self._generate_annotation_entry_data(
                            a_type, a_args, sent
                        )
                if link_types is not None:
                    for a_type, a_args in link_types.items():
                        data[a_type] = self._generate_link_entry_data(
                            a_type, a_args, sent
                        )

                if group_types is not None:
                    for a_type, a_args in group_types.items():
                        pass

                yield data

    def _process_request_args(self, a_type, a_args):

        # check the existence of ``a_type`` annotation in ``doc``
        a_meta = self.internal_metas.get(a_type)
        if a_meta is None:
            raise AttributeError(
                f"Document '{self.meta.doc_id}' has no '{a_type}' "
                f"annotations'"
            )

        # request which fields generated by which component
        component = None
        fields = {}
        if isinstance(a_args, dict):
            component = a_args.get("component")
            a_args = a_args.get("fields", {})

        if isinstance(a_args, Iterable):
            fields = set(a_args)
        elif a_args is not None:
            raise TypeError(
                f"Invalid request for '{a_type}'. "
                f"The request should be of an iterable type or a dict."
            )

        if component is None:
            component = a_meta.default_component

        if component not in a_meta.fields_created.keys():
            raise AttributeError(
                f"DataPack has no {a_type} annotations generated"
                f" by {component}"
            )

        return component, fields

    def _generate_annotation_entry_data(
            self,
            a_type: str,
            a_args: Union[Dict, List],
            sent: Optional[BaseOntology.Sentence]) -> Dict:

        component, fields = self._process_request_args(a_type, a_args)

        a_dict = dict()

        a_dict["span"] = []
        a_dict["text"] = []
        for field in fields:
            a_dict[field] = []

        sent_begin = sent.span.begin if sent else 0

        # ``a_type`` annotations generated by ``component`` in this ``sent``
        # valid_annotation = (self.index.type_index[a_type]
        #                     & self.index.component_index[component])
        # if sent:
        #     valid_annotation &= self.index.sentence_index[sent.tid]
        if sent:
            valid_annotation = self.index.sentence_index[sent.tid]
            valid_annotation &= self.index.type_index[a_type]
            valid_annotation &= self.index.component_index[component]
        else:
            valid_annotation = self.index.type_index[a_type]
            valid_annotation &= self.index.component_index[component]

        # find the lowest index (valid_begin_index) where the span.begin element
        # in the annotation is equal to the sent.span.begin
        anno_begin_index, anno_end_index = 0, len(self.annotations)

        while anno_begin_index < anno_end_index:
            mid = anno_begin_index + (anno_end_index - anno_begin_index) // 2
            if self.annotations[mid].span.begin >= sent.span.begin:
                anno_end_index = mid
            else:
                anno_begin_index = mid + 1

        valid_begin_index = anno_begin_index

        # find the largest index (valid_end_index) where the span.begin element
        # in the annotation is equal to the sent.span.end
        anno_begin_index, anno_end_index = valid_begin_index, len(self.annotations)

        while anno_begin_index < anno_end_index:
            mid = anno_begin_index + (anno_end_index - anno_begin_index) // 2
            if self.annotations[mid].span.end > sent.span.end:
                anno_end_index = mid
            else:
                anno_begin_index = mid + 1

        valid_end_index = anno_begin_index

        for annoid in range(valid_begin_index, valid_end_index):
            annotation = self.annotations[annoid]
            if annotation.tid not in valid_annotation:
                # this may be the annotation in different levels,
                # like sentence and word levels.
                continue

            a_dict["span"].append((annotation.span.begin - sent_begin,
                                   annotation.span.end - sent_begin))
            a_dict["text"].append(self.text[annotation.span.begin:
                                            annotation.span.end])
            for field in fields:
                if field not in self.internal_metas[a_type].fields_created[
                    component
                ]:
                    raise AttributeError(
                        f"'{a_type}' annotation generated by "
                        f"'{component}' has no field named '{field}'"
                    )
                a_dict[field].append(getattr(annotation, field))

        for key, value in a_dict.items():
            a_dict[key] = np.array(value)

        return a_dict

    def _generate_link_entry_data(
            self,
            a_type: str,
            a_args: Union[Dict, List],
            sent: Optional[BaseOntology.Sentence],
    ) -> Dict:

        component, fields = self._process_request_args(a_type, a_args)

        parent_fields = {f for f in fields if f.split('.')[0] == "parent"}
        child_fields = {f for f in fields if f.split('.')[0] == "child"}

        a_dict = dict()
        for field in fields:
            a_dict[field] = []
        if parent_fields:
            a_dict["parent.span"] = []
            a_dict["parent.text"] = []
        if child_fields:
            a_dict["child.span"] = []
            a_dict["child.text"] = []

        sent_begin = sent.span.begin if sent else 0

        # ``a_type`` annotations generated by ``component`` in this ``sent``
        valid_link = (self.index.type_index[a_type]
                      & self.index.component_index[component])

        if sent:
            valid_link &= self.index.sentence_index[sent.tid]

        for link in self.links:
            if link.tid not in valid_link:
                continue

            if parent_fields:
                p_id = link.parent
                parent = self.index.entry_index[p_id]
                if not isinstance(parent, Annotation):
                    raise TypeError(f"'parent'' should be an Annotation object "
                                    f"but got {type(parent)}.")
                p_type = parent.__class__.__name__
                a_dict["parent.span"].append((parent.span.begin - sent_begin,
                                              parent.span.end - sent_begin,))
                a_dict["parent.text"].append(self.text[parent.span.begin:
                                                       parent.span.end])
                for field in parent_fields:
                    p_field = field.split(".")
                    if len(p_field) == 1:
                        continue
                    if len(p_field) > 2:
                        raise AttributeError(
                            f"Too many delimiters in field name {field}."
                        )
                    p_field = p_field[1]

                    if p_field not in \
                            self.internal_metas[p_type].fields_created[
                                parent.component
                            ]:
                        raise AttributeError(
                            f"'{p_type}' annotation generated by "
                            f"'{parent.component}' has no field named "
                            f"'{p_field}'."
                        )
                    a_dict[field].append(getattr(parent, p_field))

            if child_fields:
                c_id = link.child
                child = self.index.entry_index[c_id]
                if not isinstance(child, Annotation):
                    raise TypeError(f"'parent'' should be an Annotation object "
                                    f"but got {type(child)}.")
                c_type = child.__class__.__name__
                a_dict["child.span"].append((child.span.begin - sent_begin,
                                             child.span.end - sent_begin))
                a_dict["child.text"].append(self.text[child.span.begin:
                                                      child.span.end])
                for field in child_fields:
                    c_field = field.split(".")
                    if len(c_field) == 1:
                        continue
                    if len(c_field) > 2:
                        raise AttributeError(
                            f"Too many delimiters in field name {field}."
                        )
                    c_field = c_field[1]

                    if c_field not in \
                            self.internal_metas[c_type].fields_created[
                                child.component
                            ]:
                        raise AttributeError(
                            f"'{c_type}' annotation generated by "
                            f"'{child.component}' has no field named "
                            f"'{c_field}'."
                        )
                    a_dict[field].append(getattr(child, c_field))

            for field in fields - parent_fields - child_fields:
                if field not in self.internal_metas[a_type].fields_created[
                    component
                ]:
                    raise AttributeError(
                        f"'{a_type}' annotation generated by "
                        f"'{component}' has no field named '{field}'"
                    )
                a_dict[field].append(getattr(link, field))

        for key, value in a_dict.items():
            a_dict[key] = np.array(value)
        return a_dict
