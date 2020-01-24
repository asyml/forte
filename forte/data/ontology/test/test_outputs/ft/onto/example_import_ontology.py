# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""


Automatically generated ontology . Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from typing import Optional


__all__ = [
    "Token",
    "EntityMention",
]


class Token(Annotation):
    """
    Base parent token entry

    Attributes:
        pos (Optional[str])
        lemma (Optional[str])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos: Optional[str] = None
        self.lemma: Optional[str] = None

    @property
    def pos(self):
        return self.pos

    @pos.setter
    def pos(self, pos: Optional[str]):
        self.set_fields(pos=pos)

    @property
    def lemma(self):
        return self.lemma

    @lemma.setter
    def lemma(self, lemma: Optional[str]):
        self.set_fields(lemma=lemma)


class EntityMention(Annotation):
    """

    Attributes:
        entity_type (Optional[str])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.entity_type: Optional[str] = None

    @property
    def entity_type(self):
        return self.entity_type

    @entity_type.setter
    def entity_type(self, entity_type: Optional[str]):
        self.set_fields(entity_type=entity_type)
