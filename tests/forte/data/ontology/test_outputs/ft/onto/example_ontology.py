# ***automatically_generated***
# ***source json:/Users/hector/Documents/projects/forte/tests/forte/data/ontology/test_specs/example_ontology.json***
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
        string_features (List[str]):	To demonstrate the composite type, List.
        word_forms (FList['Word']):	To demonstrate that an attribute can be a List of other entries.
        token_ranks (FDict[str, 'Word']):	To demonstrate that an attribute can be a Dict, and the values can be other entries.
    """

    string_features: List[str]
    word_forms: FList['Word']
    token_ranks: FDict[str, 'Word']

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.string_features: List[str] = []
        self.word_forms: FList['Word'] = FList(self)
        self.token_ranks: FDict[str, 'Word'] = FDict(self)


@dataclass
class WordLink(Link):
    """
    Attributes:
        string_features (List[str]):	To demonstrate the composite type, List.
    """

    string_features: List[str]

    ParentType = Word
    ChildType = Word

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.string_features: List[str] = []
