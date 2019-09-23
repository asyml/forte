"""
This class defines the basic ontology supported by our system
"""
from typing import Optional, Set

from forte.data.data_pack import DataPack
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
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos_tag = None


class Sentence(Annotation):
    pass


class Document(Annotation):
    pass


class EntityMention(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type = None


class PredicateArgument(Annotation):
    pass


class PredicateMention(Annotation):
    pass


class PredicateLink(Link):
    parent_type = PredicateMention
    child_type = PredicateArgument

    def __init__(self,
                 pack: DataPack,
                 parent: Optional[PredicateMention] = None,
                 child: Optional[PredicateArgument] = None):
        super().__init__(pack, parent, child)
        self.arg_type = None


class CoreferenceMention(Annotation):
    pass


class CoreferenceGroup(Group):
    member_type = CoreferenceMention

    def __init__(self,
                 pack: DataPack,
                 members: Optional[Set[CoreferenceMention]] = None):
        super().__init__(pack, members)  # type: ignore
        self.coref_type = None
