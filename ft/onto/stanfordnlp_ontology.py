# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology . Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Link
from typing import Optional


__all__ = [
    "Token",
    "Sentence",
    "Document",
    "Dependency",
]


class Token(Annotation):
    """
    Attributes:
        _lemma (Optional[str])
        _pos_tag (Optional[str])
        _upos (Optional[str])
        _xpos (Optional[str])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._lemma: Optional[str] = None
        self._pos_tag: Optional[str] = None
        self._upos: Optional[str] = None
        self._xpos: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['lemma'] = self._lemma
        state['pos_tag'] = self._pos_tag
        state['upos'] = self._upos
        state['xpos'] = self._xpos
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._lemma = state.get('lemma', None) 
        self._pos_tag = state.get('pos_tag', None) 
        self._upos = state.get('upos', None) 
        self._xpos = state.get('xpos', None) 

    @property
    def lemma(self):
        return self._lemma

    @lemma.setter
    def lemma(self, lemma: Optional[str]):
        self.set_fields(_lemma=lemma)

    @property
    def pos_tag(self):
        return self._pos_tag

    @pos_tag.setter
    def pos_tag(self, pos_tag: Optional[str]):
        self.set_fields(_pos_tag=pos_tag)

    @property
    def upos(self):
        return self._upos

    @upos.setter
    def upos(self, upos: Optional[str]):
        self.set_fields(_upos=upos)

    @property
    def xpos(self):
        return self._xpos

    @xpos.setter
    def xpos(self, xpos: Optional[str]):
        self.set_fields(_xpos=xpos)


class Sentence(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Document(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Dependency(Link):
    """
    Attributes:
        _rel_type (Optional[str])
    """
    ParentType = Token

    ChildType = Token

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self._rel_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['rel_type'] = self._rel_type
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._rel_type = state.get('rel_type', None) 

    @property
    def rel_type(self):
        return self._rel_type

    @rel_type.setter
    def rel_type(self, rel_type: Optional[str]):
        self.set_fields(_rel_type=rel_type)
