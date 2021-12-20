# ***automatically_generated***
# ***source json:tests/forte/data/ontology/test_specs/test_top_attribute.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology test_top_attribute. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Generics
from forte.data.ontology.top import Group
from forte.data.ontology.top import Link
from forte.data.ontology.top import MultiPackGeneric
from forte.data.ontology.top import MultiPackGroup
from forte.data.ontology.top import MultiPackLink
from forte.data.ontology.top import Query
from typing import Optional

__all__ = [
    "Item",
]


@dataclass
class Item(Annotation):
    """
    Attributes:
        query (Optional[Query]):
        generics (Optional[Generics]):
        link (Optional[Link]):
        group (Optional[Group]):
        multiPackLink (Optional[MultiPackLink]):
        multiPackGroup (Optional[MultiPackGroup]):
        multiPackGeneric (Optional[MultiPackGeneric]):
    """

    query: Optional[Query]
    generics: Optional[Generics]
    link: Optional[Link]
    group: Optional[Group]
    multiPackLink: Optional[MultiPackLink]
    multiPackGroup: Optional[MultiPackGroup]
    multiPackGeneric: Optional[MultiPackGeneric]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.query: Optional[Query] = None
        self.generics: Optional[Generics] = None
        self.link: Optional[Link] = None
        self.group: Optional[Group] = None
        self.multiPackLink: Optional[MultiPackLink] = None
        self.multiPackGroup: Optional[MultiPackGroup] = None
        self.multiPackGeneric: Optional[MultiPackGeneric] = None
