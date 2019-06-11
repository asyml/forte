""" This class defines the basic ontology supported by our system
"""
from abc import abstractmethod


class TOP:
    """The base class inherited by all annotations

    Args:
        component (str): the engine name that creates this annotation,
            it should at least include the full engine name.
        tid (str, optional): a doc level universal id to identify
            this annotation
    """
    def __init__(self, component: str, tid: str = None):
        self.tid = f"{self.__class__.__qualname__}.{tid}"
        self.component = component

    def set_tid(self, tid: str):
        self.tid = f"{self.__class__.__qualname__}.{tid}"

    @abstractmethod
    def hash(self):
        pass

    @abstractmethod
    def eq(self, other):
        pass

    def __hash__(self):
        return self.hash()

    def __eq__(self, other):
        return self.eq(other)


class Span:
    def __init__(self, begin: int, end: int):
        self.begin = begin
        self.end = end


class Annotation(TOP):
    def __init__(self, component: str, span: Span, tid: str = None):
        super().__init__(component, tid)
        self.span = span

    def hash(self):
        return hash(
            (
                self.span.begin,
                self.span.end,
                self.component,
                type(self)
            )
        )

    def eq(self, other):
        return (
            (type(self) == type(other))
            and (self.component == other.component)
            and (self.span.begin == other.span.begin)
            and (self.span.end == other.span.end)
        )


class Link(TOP):
    def __init__(self, component: str, tid: str = None):
        super().__init__(component, tid)
        self.parent = None
        self.child = None

    def set_parent(self, parent: TOP):
        self.parent = parent

    def set_child(self, child: TOP):
        self.child = child

    def hash(self):
        return hash(
            (self.parent,
             self.child,
             self.component,
             type(self))
        )

    def eq(self, other):
        return (
            (type(self) == type(other))
            and (self.component == other.component)
            and (self.parent == other.parent)
            and (self.child == other.child)
        )


class Group(TOP):
    def __init__(self, component: str, tid: str = None):
        super().__init__(component, tid)
        self.members = set()

    def add_member(self, member: TOP):
        self.members.add(member)

    def eq(self, other):
        return (
            (type(self) == type(other))
            and (self.members == other.members)
        )


class Meta:
    def __init__(self, docid: str = None):
        self.docid = docid


class BaseOntology:
    class Token(Annotation):
        def __init__(self, component: str, span: Span, tid: str = None):
            super().__init__(component, span, tid)

    class Sentence(Annotation):
        def __init__(self, component: str, span: Span, tid: str = None):
            super().__init__(component, span, tid)

    class EntityMention(Annotation):
        def __init__(self, component: str, span: Span, tid: str = None):
            super().__init__(component, span, tid)
            self.ner_type = None

    class PredicateArgument(Annotation):
        def __init__(self,  component: str, span: Span, tid: str = None):
            super().__init__(component, span, tid)

    class PredicateLink(Link):
        def __init__(self, component: str, tid: str = None):
            super().__init__(component, tid)
            self.arg_type = None

        def set_argtype(self, arg_type: str):
            self.arg_type = arg_type

    class PredicateMention(Annotation):
        def __init__(self, component: str, span: Span, tid: str = None):
            super().__init__(component, span, tid)
            self.links = []

        def add_link(self, link: Link):
            self.links.append(link)

    class CoreferenceGroup(Group):
        def __init__(self, component: str, tid: str = None):
            super().__init__(component, tid)
            self.coref_type = None

        def eq(self, other):
            return (
                (type(self) == type(other))
                and (self.coref_type == other.coref_type)
                and (self.members == other.members)
            )

    class CoreferenceMention(Annotation):
        def __init__(self, component: str, span: Span, tid: str = None):
            super().__init__(component, span, tid)
