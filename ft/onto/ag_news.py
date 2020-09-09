# ***automatically_generated***
# ***source json:../../../../../../capstone/forte/forte/ontology_specs/ag_news.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology ag_news. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from ft.onto.base_ontology import Document
from typing import Optional

__all__ = [
    "Article",
    "Title",
    "Description",
]


@dataclass
class Article(Document):
    """
    Attributes:
        class_id (Optional[int])	The class ids are numbered 1-4 where 1 represents World, 2 represents Sports, 3 represents Business and 4 represents Sci/Tech.
        class_name (Optional[str])	The name of the class.
    """

    class_id: Optional[int]
    class_name: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.class_id: Optional[int] = None
        self.class_name: Optional[str] = None


@dataclass
class Title(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class Description(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
