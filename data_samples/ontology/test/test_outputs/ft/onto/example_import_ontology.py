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
        _pos (Optional[str])
        _lemma (Optional[str])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._pos: Optional[str] = None
        self._lemma: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['pos'] = self._pos
        state['lemma'] = self._lemma
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._pos = state.get('pos', None) 
        self._lemma = state.get('lemma', None) 

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos: Optional[str]):
        self.set_fields(_pos=pos)

    @property
    def lemma(self):
        return self._lemma

    @lemma.setter
    def lemma(self, lemma: Optional[str]):
        self.set_fields(_lemma=lemma)


class EntityMention(Annotation):
    """

    Attributes:
        _entity_type (Optional[str])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._entity_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['entity_type'] = self._entity_type
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._entity_type = state.get('entity_type', None) 

    @property
    def entity_type(self):
        return self._entity_type

    @entity_type.setter
    def entity_type(self, entity_type: Optional[str]):
        self.set_fields(_entity_type=entity_type)
