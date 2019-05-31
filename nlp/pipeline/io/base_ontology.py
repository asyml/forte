""" This class defines the basic ontology of supported by our system
"""
from abc import ABCMeta, abstractmethod


class Ontology:
    """
    The ontology class that is used to host all ontology definitions.
    """
    def __init__(self):
        pass

    def add_ontology_type(self, clz: type):
        if not clz == TOP:
            raise ValueError('Illegal ontology class type. All '
                             'class types should be subclass of '
                             'nlp.pipeline.io.base_ontology.TOP')


class TOP:
    def __init__(self, tid: str, component: str, is_gold: bool = False):
        # tid is doc level universal id to identify this annotation, e.g.
        # "entity_mention_12"
        self.tid = tid

        # component is the engine name that creates this
        # annotation, it should at least include the full engine name
        # e.g. "nlp.pipeline.engines.ner", but can be more specific.
        self.component = component

        # this field is used to indicate whether it can be used as a gold
        # standard. I am still thinking whether to use it.
        self.is_gold = is_gold

    @abstractmethod
    def hash(self):
        pass

    @abstractmethod
    def eq(self):
        pass

    def __hash__(self):
        return self.hash()

    def __eq__(self, other):
        return self.eq()


class Span:
    def __init__(self, begin: int, end: int):
        self.begin = begin
        self.end = end


class Annotation(TOP):
    def __init__(self, tid: str, component: str, span: Span):
        super().__init__(tid, component)
        self.span = span


class Link(TOP):
    def __init__(self, tid: str, component: str):
        super().__init__(tid, component)
        self.parent = None
        self.child = None

    def set_parent(self, parent: TOP):
        self.parent = parent

    def set_child(self, child: TOP):
        self.child = child


class Group(TOP):
    def __init__(self, tid: str, component: str):
        super().__init__(tid, component)
        self.members = set()

    def add_member(self, member: TOP):
        self.members.add(member)
