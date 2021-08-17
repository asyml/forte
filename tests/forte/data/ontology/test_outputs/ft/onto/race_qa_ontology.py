# ***automatically_generated***
# ***source json:../../../../../../Documents/forte_develop/forte/tests/forte/data/ontology/test_specs/race_qa_onto.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology race_qa_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.core import FList
from forte.data.ontology.top import Annotation
from ft.onto.base_ontology import Document
from typing import List
from typing import Optional

__all__ = [
    "RaceDocument",
    "Passage",
    "Option",
    "Question",
]


@dataclass
class RaceDocument(Document):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class Passage(Document):
    """
    Attributes:
        passage_id (Optional[str]):
    """

    passage_id: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.passage_id: Optional[str] = None


@dataclass
class Option(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class Question(Annotation):
    """
    Attributes:
        options (FList[Option]):
        answers (List[int]):
    """

    options: FList[Option]
    answers: List[int]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.options: FList[Option] = FList(self)
        self.answers: List[int] = []
