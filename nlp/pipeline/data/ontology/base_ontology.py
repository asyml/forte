"""
This class defines the basic ontology supported by our system
"""
from typing import Optional, Set
from nlp.pipeline.data.ontology.top import *

__all__ = [
    "Token",
    "EntityMention",
    "Sentence",
    "PredicateMention",
    "PredicateLink",
    "PredicateArgument",
    "CoreferenceGroup",
    "CoreferenceMention"
]


class Token(Annotation):
    def __init__(self, component: str, begin: int, end: int):
        super().__init__(component, begin, end)
        self.pos_tag = None


class Sentence(Annotation):
    pass


class EntityMention(Annotation):
    def __init__(self, component: str, begin: int, end: int):
        super().__init__(component, begin, end)
        self.ner_type = None


class PredicateArgument(Annotation):
    pass


class PredicateMention(Annotation):
    pass


class PredicateLink(Link):
    parent_type = PredicateMention
    child_type = PredicateArgument

    def __init__(self, component: str,
                 parent: Optional[parent_type] = None,
                 child: Optional[child_type] = None):
        super().__init__(component, parent, child)
        self.arg_type = None


class CoreferenceMention(Annotation):
    pass


class CoreferenceGroup(Group):
    member_type = CoreferenceMention

    def __init__(self, component: str,
                 members: Optional[Set[member_type]] = None):
        super().__init__(component, members)
        self.coref_type = None


