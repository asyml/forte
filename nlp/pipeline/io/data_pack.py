""" This class defines the core interchange format, deals with basic operations
such as reading, writing, checking and indexing.
"""
import logging
import itertools
from collections import defaultdict
from typing import Union
from nlp.pipeline.io.base_ontology import *
from sortedcontainers import SortedList

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class InternalMeta:
    def __init__(self):
        self.id_counter = 0
        self.fields_created = defaultdict(set)
        self.default_component = None


class DataIndex:
    def __init__(self):
        self.entry_index = defaultdict(Entry)
        self.type_index = defaultdict(set)
        self.sentence_index = defaultdict(set)
        self.component_index = defaultdict(set)


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
                entry.set_tid(str(self.internal_metas[name].id_counter)),
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

                # sentence indexing: time complexity could be improved by bisect
                if isinstance(entry, BaseOntology.Sentence):
                    for prev_entry in itertools.chain(self.annotations,
                                                      self.links,
                                                      self.groups):
                        if self._in_span(prev_entry, entry.span):
                            self.index.sentence_index[entry.tid].add(
                                prev_entry.tid)
                else:
                    for sent_id in self.index.sentence_index.keys():
                        sent = self.index.entry_index[sent_id]
                        if hasattr(sent, "span") and \
                                self._in_span(entry, sent.span):
                            self.index.sentence_index[sent_id].add(
                                entry.tid)
            return entry.tid
        else:
            # logger.debug(f"Annotation already exist {annotation.tid}")
            return target[target.index(entry)].tid

    def record_fields(self, fields: list, component: str, entry_type: str):
        """Record in the internal meta that ``component`` has generated
        ``fields`` for ``entry_type``.
        """
        if entry_type not in self.internal_metas.keys():
            self.internal_metas[entry_type].default_component = component
        for f in fields:
            self.internal_metas[entry_type].fields_created[component].add(f)

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
        for entry in itertools.chain(self.annotations, self.links, self.groups):
            for sent_id in self.index.type_index["Sentence"]:
                sent = self.index.entry_index[sent_id]
                if hasattr(sent, "span") and self._in_span(entry, sent.span):
                    self.index.sentence_index[sent_id].add(entry.tid)

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
                m = self.index.entry_index.get(m_id)
                if inner_begin == -1:
                    inner_begin = m.span.begin
                inner_begin = min(inner_begin, m.span.begin)
                inner_end = min(inner_end, m.span.end)
        else:
            raise ValueError(
                f"Invalid entry type {type(inner_entry)}. A valid entry "
                f"should be an instance of Annotation, Link, or Group."
            )
        return inner_begin >= span.begin and inner_end <= span.end
