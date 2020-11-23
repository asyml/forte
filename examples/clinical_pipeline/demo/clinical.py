# ***automatically_generated***
# ***source json:examples/clinical_pipeline/clinical_onto.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology clinical. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from ft.onto.base_ontology import EntityMention

__all__ = [
    "ClinicalEntityMention",
    "Description",
    "Body",
]


@dataclass
class ClinicalEntityMention(EntityMention):
    """
    A span based annotation `ClinicalEntityMention`, normally used to represent an Entity Mention in a piece of clinical text.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class Description(Annotation):
    """
    A span based annotation `Description`, used to represent the description in a piece of clinical note.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class Body(Annotation):
    """
    A span based annotation `Body`, used to represent the actual content in a piece of clinical note.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
