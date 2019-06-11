""" This class defines the core interchange format, deals with basic operations
such as reading, writing, checking and indexing.
"""
import logging
from collections import defaultdict
from nlp.pipeline.io.base_ontology import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class InternalMeta:
    def __init__(self):
        self.id_counter = 0
        self.fields_created = defaultdict(set)


class AnnotationIndex:
    def __init__(self):
        self.annotation_index = defaultdict(TOP)
        self.type_index = defaultdict(set)


class Interchange:
    def __init__(self):
        self.annotations = []
        self.meta = Meta()
        self.index = AnnotationIndex()
        self.internal_metas = defaultdict(InternalMeta)

    def add_annotation(self, annotation: TOP, indexing: bool = True):
        """
        Try to add an annotation to the Interchange object. If a same
        annotation already exists, will not add the new one.

        Returns:
            If a same annotation already exists, returns the tid of the
            existing annotation. Otherwise, return the tid of the annotation
            just added.
        """
        if annotation not in self.annotations:
            qual_name = annotation.__class__.__qualname__
            annotation.set_tid(str(self.internal_metas[qual_name].id_counter)),
            self.annotations.append(annotation)
            self.internal_metas[qual_name].id_counter += 1
            if indexing:
                self.index.annotation_index[annotation.tid] = annotation
                self.index.type_index[qual_name].add(annotation.tid)
            return annotation.tid
        else:
            logger.debug(f"Annotation already exist {annotation.tid}")
            return self.annotations[self.annotations.index(annotation)].tid

    def add_fields(self, fields: list, component: str, annotation_type: str):
        for f in fields:
            internal_meta = self.internal_metas[annotation_type]
            internal_meta.fields_created[component].add(f)

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.meta, k):
                raise AttributeError(f"Meta has no attribute {k}")
            setattr(self.meta, k, v)

    def index_annotations(self):
        for annotation in self.annotations:
            qual_name = annotation.__class__.__qualname__
            self.index.annotation_index[annotation.tid] = annotation
            self.index.type_index[qual_name].add(annotation.tid)
