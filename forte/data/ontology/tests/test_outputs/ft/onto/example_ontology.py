# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology . Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
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
        _string_features (Optional[List[str]])	To demonstrate the composite type, List.
        _word_forms (Optional[List[int]])	To demonstrate that an attribute can be a List of other entries.
        _token_ranks (Optional[Dict[int, int]])	To demonstrate that an attribute can be a Dict, and the values can be other entries.
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._string_features: Optional[List[str]] = []
        self._word_forms: Optional[List[int]] = []
        self._token_ranks: Optional[Dict[int, int]] = {}

    def __getstate__(self): 
        state = super().__getstate__()
        state['string_features'] = self._string_features
        state['word_forms'] = self._word_forms
        state['token_ranks'] = self._token_ranks
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._string_features = state.get('string_features', None) 
        self._word_forms = state.get('word_forms', None) 
        self._token_ranks = state.get('token_ranks', None) 

    @property
    def string_features(self):
        return self._string_features

    @string_features.setter
    def string_features(self, string_features: Optional[List[str]]):
        string_features = [] if string_features is None else string_features
        self.set_fields(_string_features=string_features)

    def num_string_features(self):
        return len(self._string_features)

    def clear_string_features(self):
        self._string_features.clear()

    def add_string_features(self, a_string_features: str):
        self._string_features.append(a_string_features)

    @property
    def word_forms(self):
        return [self.pack.get_entry(tid) for tid in self._word_forms]

    @word_forms.setter
    def word_forms(self, word_forms: Optional[List["Word"]]):
        word_forms = [] if word_forms is None else word_forms
        self.set_fields(_word_forms=[self.pack.add_entry_(obj) for obj in word_forms])

    def num_word_forms(self):
        return len(self._word_forms)

    def clear_word_forms(self):
        [self.pack.delete_entry(self.pack.get_entry(tid)) for tid in self._word_forms]
        self._word_forms.clear()

    def add_word_forms(self, a_word_forms: "Word"):
        self._word_forms.append(self.pack.add_entry_(a_word_forms))

    @property
    def token_ranks(self):
        return {self.pack.get_entry(self._token_ranks[key]) for key in self._token_ranks}

    @token_ranks.setter
    def token_ranks(self, token_ranks: Optional[Dict[int, "Word"]]):
        token_ranks = {} if token_ranks is None else token_ranks
        self.set_fields(_token_ranks=dict([(k, self.pack.add_entry_(v)) for k, v in token_ranks.items()]))

    def num_token_ranks(self):
        return len(self._token_ranks)

    def clear_token_ranks(self):
        [self.pack.delete_entry(self.pack.get_entry(tid)) for tid in self.token_ranks.values()]
        self._token_ranks.clear()

    def add_token_ranks(self, key: int, value: "Word"):
        self._token_ranks[key] = self.pack.add_entry_(value)


class WordLink(Link):
    """
    Attributes:
        _string_features (Optional[List[str]])	To demonstrate the composite type, List.
    """
    ParentType = Word

    ChildType = Word

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self._string_features: Optional[List[str]] = []

    def __getstate__(self): 
        state = super().__getstate__()
        state['string_features'] = self._string_features
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self._string_features = state.get('string_features', None) 

    @property
    def string_features(self):
        return self._string_features

    @string_features.setter
    def string_features(self, string_features: Optional[List[str]]):
        string_features = [] if string_features is None else string_features
        self.set_fields(_string_features=string_features)

    def num_string_features(self):
        return len(self._string_features)

    def clear_string_features(self):
        self._string_features.clear()

    def add_string_features(self, a_string_features: str):
        self._string_features.append(a_string_features)
