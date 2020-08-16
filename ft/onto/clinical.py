# ***automatically_generated***
# ***source json:forte/ontology_specs/clinical.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology clinical. Do not change manually.
"""

from dataclasses import dataclass
from typing import Optional

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Generics

__all__ = [
    "ClinicalEntityMention",
]


@dataclass
class ClinicalEntityMention(Annotation):
    """
    A span based annotation `ClinicalEntityMention`, normally used to represent an Entity Mention in a piece of clinical text.
    Attributes:
        ner_type (Optional[str])
    """

    ner_type: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: Optional[str] = None
