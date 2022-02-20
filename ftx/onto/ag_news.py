# ***automatically_generated***
# ***source json:../../../../casl/forte/forte/ontology_specs/ag_news.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology ag_news. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation

__all__ = [
    "Description",
]


@dataclass
class Description(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
