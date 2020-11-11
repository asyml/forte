# ***automatically_generated***
# ***source json:tests/forte/data/ontology/test_specs/example_complex_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology example_complex_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from forte.data.ontology.core import FList
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Link
from typing import Optional

__all__ = [
    "Token",
    "Sentence",
    "Document",
    "Dependency",
]


@dataclass
class Token(Annotation):
    """
    Attributes:
        lemma (Optional[str])
        is_verb (Optional[bool])
        num_chars (Optional[int])
        score (Optional[float])
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
        key_tokens (FList[Token])
    """

    key_tokens: FList[Token]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.key_tokens: FList[Token] = FList(self)


@dataclass
class Document(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class Dependency(Link):
    """
    Attributes:
        rel_type (Optional[str])
    """

    rel_type: Optional[str]

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.rel_type: Optional[str] = None
