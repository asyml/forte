# ***automatically_generated***
# ***source json:tests/forte/data/ontology/test_specs/example_multi_module_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology example_multi_module_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.core import FList
from forte.data.ontology.top import Annotation
from typing import Optional

__all__ = [
    "Token",
    "Sentence",
    "Document",
]


@dataclass
class Token(Annotation):
    """
    Attributes:
        lemma (Optional[str]):
        is_verb (Optional[bool]):
        num_chars (Optional[int]):
        score (Optional[float]):
    """

    lemma: Optional[str]
    is_verb: Optional[bool]
    num_chars: Optional[int]
    score: Optional[float]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.lemma: Optional[str] = None
        self.is_verb: Optional[bool] = None
        self.num_chars: Optional[int] = None
        self.score: Optional[float] = None


@dataclass
class Sentence(Annotation):
    """
    Attributes:
        tokens (FList[Token]):
    """

    tokens: FList[Token]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.tokens: FList[Token] = FList(self)


@dataclass
class Document(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
