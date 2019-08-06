"""
This class defines the basic ontology supported by our system
"""
from typing import Optional, Set
from nlp.pipeline.data.ontology.top import Annotation, Link, Group

__all__ = [
    "Token",
    "Document",
    "EntityMention",
    "Sentence",
    "PredicateMention",
    "PredicateLink",
    "PredicateArgument",
    "CoreferenceGroup",
    "CoreferenceMention"
]


class Token(Annotation):
    """
    A span based annotation :class:`Token`.
    """
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.pos_tag = None


class Sentence(Annotation):
    """
    A span based annotation :class:`Sentence`.
    """
    pass


class Document(Annotation):
    """
    A span based annotation :class:`Document`.
    """
    pass


class EntityMention(Annotation):
    """
    A span based annotation :class:`EntityMention`.
    """
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.ner_type = None


class PredicateArgument(Annotation):
    """
    A span based annotation :class:`PredicateArgument`.
    """
    pass


class PredicateMention(Annotation):
    """
    A span based annotation :class:`PredicateMention`.
    """
    pass


class PredicateLink(Link):
    """
    A :class:`Link` type entry which take :class:`PredicateMention` as parent
    and :class:`PredicateArgument` as child.
    """
    parent_type = PredicateMention
    child_type = PredicateArgument

    def __init__(self,
                 parent: Optional[PredicateMention] = None,
                 child: Optional[PredicateArgument] = None):
        super().__init__(parent, child)
        self.arg_type = None


class CoreferenceMention(Annotation):
    """
    A span based annotation :class:`CoreferenceMention`.
    """
    pass


class CoreferenceGroup(Group):
    """
    A :class:`Group` type entry which take :class:`CoreferenceMention` as
    members.
    """
    member_type = CoreferenceMention

    def __init__(self, members: Optional[Set[CoreferenceMention]] = None):
        super().__init__(members)  # type: ignore
        self.coref_type = None
