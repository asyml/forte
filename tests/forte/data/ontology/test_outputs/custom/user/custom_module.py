# ***automatically_generated***
# ***source json:/Users/hector/Documents/projects/forte/tests/forte/data/ontology/test_specs/example_multi_module_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology example_multi_module_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Link
from ft.onto.ft_module import Token
from typing import Optional

__all__ = [
    "Dependency",
]


@dataclass
class Dependency(Link):
    """
    Attributes:
        rel_type (Optional[str]):
    """

    rel_type: Optional[str]

    ParentType = Token
    ChildType = Token

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.rel_type: Optional[str] = None
