# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""


Automatically generated ontology . Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Link
from ft.onto.example_import_ontology import Token
from typing import Dict
from typing import List
from typing import Optional


__all__ = [
    "Word",
    "WordLink",
]


class Word(Token):
    """

    Attributes:
        string_features (Optional[List[str]])	To demonstrate the composite type, List.
        word_forms (Optional[List["Word"]])	To demonstrate that an attribute can have the entry it is contained in as it's type or item_type.
        token_ranks (Optional[Dict[int, "Word"]])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.string_features: Optional[List[str]] = []
        self.word_forms: Optional[List["Word"]] = []
        self.token_ranks: Optional[Dict[int, "Word"]] = None

    @property
    def string_features(self):
        return self.string_features

    @string_features.setter
    def string_features(self, string_features: Optional[List[str]]):
        self.set_fields(string_features=string_features)

    def num_string_features(self):
        return len(self.string_features)

    def clear_string_features(self):
        self.string_features.clear()

    def add_string_features(self, a_string_features: str):
        self.string_features.append(a_string_features)

    @property
    def word_forms(self):
        return self.word_forms

    @word_forms.setter
    def word_forms(self, word_forms: Optional[List["Word"]]):
        self.set_fields(word_forms=word_forms)

    def num_word_forms(self):
        return len(self.word_forms)

    def clear_word_forms(self):
        self.word_forms.clear()

    def add_word_forms(self, a_word_forms: "Word"):
        self.word_forms.append(a_word_forms)

    @property
    def token_ranks(self):
        return self.token_ranks

    @token_ranks.setter
    def token_ranks(self, token_ranks: Optional[Dict[int, "Word"]]):
        self.set_fields(token_ranks=token_ranks)

    def num_token_ranks(self):
        return len(self.token_ranks)

    def clear_token_ranks(self):
        self.token_ranks.clear()

    def add_token_ranks(self, key: int, value: "Word"):
        self.token_ranks[key](value)


class WordLink(Link):
    """

    Attributes:
        string_features (Optional[List[str]])	To demonstrate the composite type, List.

    """

    ParentType = Word

    ChildType = Word

    def __init__(self, pack: DataPack, parent: Optional[DataPack] = None, child: Optional[DataPack] = None):
        super().__init__(pack, parent, child)
        self.string_features: Optional[List[str]] = []

    @property
    def string_features(self):
        return self.string_features

    @string_features.setter
    def string_features(self, string_features: Optional[List[str]]):
        self.set_fields(string_features=string_features)

    def num_string_features(self):
        return len(self.string_features)

    def clear_string_features(self):
        self.string_features.clear()

    def add_string_features(self, a_string_features: str):
        self.string_features.append(a_string_features)
