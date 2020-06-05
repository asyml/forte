# ***automatically_generated***
# ***source json:tests/forte/data/ontology/test_specs/example_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology example_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from forte.data.ontology.core import FDict
from forte.data.ontology.core import FList
from forte.data.ontology.top import Link
from ft.onto.example_import_ontology import Token
from typing import List
from typing import Optional

__all__ = [
    "Word",
    "WordLink",
]


@dataclass
class Word(Token):
    """
    Attributes:
        string_features (Optional[List[str]])	To demonstrate the composite type, List.
        word_forms (Optional[FList[Word]])	To demonstrate that an attribute can be a List of other entries.
        token_ranks (Optional[FDict[int, "Word"]])	To demonstrate that an attribute can be a Dict, and the values can be other entries.
    """

    string_features: Optional[List[str]]
    word_forms: Optional[FList[Word]]
    token_ranks: Optional[FDict[int, "Word"]]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.string_features: Optional[List[str]] = List[str]()
        self.word_forms: Optional[FList[Word]] = FList[Word]()
        self.token_ranks: Optional[FDict[int, "Word"]] = FDict[int, "Word"]()


@dataclass
class WordLink(Link):
    """
    Attributes:
        string_features (Optional[List[str]])	To demonstrate the composite type, List.
    """

    string_features: Optional[List[str]]

    ParentType = Word
    ChildType = Word

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.string_features: Optional[List[str]] = List[str]()
