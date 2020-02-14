# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""


Automatically generated ontology . Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Link
from ft.onto.ft_module import Token
from typing import Optional


__all__ = [
    "Dependency",
]


class Dependency(Link):
    """

    Attributes:
        rel_type (Optional[str])

    """

    ParentType = Token

    ChildType = Token

    def __init__(self, pack: DataPack, parent: Optional[DataPack] = None, child: Optional[DataPack] = None):
        super().__init__(pack, parent, child)
        self.rel_type: Optional[str] = None

    @property
    def rel_type(self):
        return self.rel_type

    @rel_type.setter
    def rel_type(self, rel_type: Optional[str]):
        self.set_fields(rel_type=rel_type)
