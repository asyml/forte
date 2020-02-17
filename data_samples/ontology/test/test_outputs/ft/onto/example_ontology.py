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
        string_features (Optional[List[int]])	To demonstrate the composite type, List.
        word_forms (Optional[List[int]])	To demonstrate that an attribute can be a List of other entries.
        token_ranks (Optional[Dict[int, int]])	To demonstrate that an attribute can be a Dict, and the values can be other entries.

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.string_features: Optional[List[int]] = []
        self.word_forms: Optional[List[int]] = []
        self.token_ranks: Optional[Dict[int, int]] = {}

    def __getstate__(self): 
        state = super().__getstate__()
        state['string_features'] = self.string_features
        state['word_forms'] = self.word_forms
        state['token_ranks'] = self.token_ranks
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.string_features = state.get('string_features', []) 
        self.word_forms = state.get('word_forms', []) 
        self.token_ranks = state.get('token_ranks', {}) 

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
        self.set_fields(word_forms=[self.__pack.add_entry_(obj) for obj in word_forms])

    def num_word_forms(self):
        return len(self.word_forms)

    def clear_word_forms(self):
        [self.__pack.delete_entry(self.__pack.get_entry(tid)) for tid in self.word_forms]
        self.word_forms.clear()

    def add_word_forms(self, a_word_forms: "Word"):
        self.word_forms.append(self.__pack.add_entry_(a_word_forms))

    @property
    def token_ranks(self):
        return self.token_ranks

    @token_ranks.setter
    def token_ranks(self, token_ranks: Optional[Dict[int, "Word"]]):
        self.set_fields(token_ranks=dict([(k, self.__pack.add_entry_(v)) for k, v in token_ranks.items()]))

    def num_token_ranks(self):
        return len(self.token_ranks)

    def clear_token_ranks(self):
        [self.__pack.delete_entry(self.__pack.get_entry(tid)) for tid in self.token_ranks.values()]
        self.token_ranks.clear()

    def add_token_ranks(self, key: int, value: "Word"):
        self.token_ranks[key] = self.__pack.add_entry_(value)


class WordLink(Link):
    """

    Attributes:
        string_features (Optional[List[int]])	To demonstrate the composite type, List.

    """

    ParentType = Word

    ChildType = Word

    def __init__(self, pack: DataPack, parent: Optional[DataPack] = None, child: Optional[DataPack] = None):
        super().__init__(pack, parent, child)
        self.string_features: Optional[List[int]] = []

    def __getstate__(self): 
        state = super().__getstate__()
        state['string_features'] = self.string_features
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.string_features = state.get('string_features', []) 

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
