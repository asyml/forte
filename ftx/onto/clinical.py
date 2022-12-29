# ***automatically_generated***
# ***source json:/Users/hector/Documents/projects/forte/forte/ontology_specs/medical.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology medical. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.core import FList
from forte.data.ontology.top import Generics
from ft.onto.base_ontology import EntityMention
from typing import List
from typing import Optional

__all__ = [
    "UMLSConceptLink",
    "MedicalEntityMention",
]


@dataclass
class UMLSConceptLink(Generics):
    """
    A umls concept entity, used to represent basic information of a umls concept
    Attributes:
        cui (Optional[str]):
        name (Optional[str]):
        definition (Optional[str]):
        tuis (List[str]):
        aliases (List[str]):
        score (Optional[str]):
    """

    cui: Optional[str]
    name: Optional[str]
    definition: Optional[str]
    tuis: List[str]
    aliases: List[str]
    score: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.cui: Optional[str] = None
        self.name: Optional[str] = None
        self.definition: Optional[str] = None
        self.tuis: List[str] = []
        self.aliases: List[str] = []
        self.score: Optional[str] = None


@dataclass
class MedicalEntityMention(EntityMention):
    """
    A span based annotation class MedicalEntityMention, used to represent an Entity Mention in medical domain
    Attributes:
        umls_link (Optional[str]):
        umls_entities (FList[UMLSConceptLink]):
    """

    umls_link: Optional[str]
    umls_entities: FList[UMLSConceptLink]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.umls_link: Optional[str] = None
        self.umls_entities: FList[UMLSConceptLink] = FList(self)
