# ***automatically_generated***
# ***source json:forte/ontology_specs/clinical.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology clinical. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Generics
from typing import Optional

__all__ = [
    "ClinicalEntityMention",
]


@dataclass
class ClinicalEntityMention(Annotation):
    """
    A span based annotation `ClinicalEntityMention`, normally used to represent an Entity Mention in a piece of clinical text.
    Attributes:
        cliner_type (Optional[str])
    """

    cliner_type: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.cliner_type: Optional[str] = None

