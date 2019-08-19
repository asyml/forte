"""
This class defines the basic ontology supported by our system
"""
from typing import Optional, Set
from forte.data.ontology.top import Annotation, Link, Group

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
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.pos_tag = None


class Sentence(Annotation):
    pass


class Document(Annotation):
    pass


class EntityMention(Annotation):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.ner_type = None


class PredicateArgument(Annotation):
    pass


class PredicateMention(Annotation):
    pass


class PredicateLink(Link):
    parent_type = PredicateMention
    child_type = PredicateArgument

    def __init__(self,
                 parent: Optional[PredicateMention] = None,
                 child: Optional[PredicateArgument] = None):
        super().__init__(parent, child)
        self.arg_type = None


class CoreferenceMention(Annotation):
    pass


class CoreferenceGroup(Group):
    member_type = CoreferenceMention

    def __init__(self, members: Optional[Set[CoreferenceMention]] = None):
        super().__init__(members)  # type: ignore
        self.coref_type = None
