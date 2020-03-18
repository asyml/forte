# ***automatically_generated***
# ***source json:tests/forte/data/ontology/test_specs/example_multi_module_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology example_multi_module_ontology. Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Link
from ft.onto.ft_module import Token
from typing import Optional


__all__ = [
    "Dependency",
]


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
        state['rel_type'] = state.pop('_rel_type')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._rel_type = state.get('rel_type', None) 

    @property
    def rel_type(self):
        return self._rel_type

    @rel_type.setter
    def rel_type(self, rel_type: Optional[str]):
        self.set_fields(_rel_type=rel_type)
