# ***automatically_generated***
# ***source json:/Users/hector/Documents/projects/forte/tests/forte/data/ontology/test_specs/example_import_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology example_import_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from typing import Optional

__all__ = [
    "Token",
    "EntityMention",
]


@dataclass
class Token(Annotation):
    """
    Base parent token entry
    Attributes:
        pos (Optional[str]):
        lemma (Optional[str]):
    """

    pos: Optional[str]
    lemma: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos: Optional[str] = None
        self.lemma: Optional[str] = None


@dataclass
class EntityMention(Annotation):
    """
    Attributes:
        entity_type (Optional[str]):
    """

    entity_type: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.entity_type: Optional[str] = None
