# ***automatically_generated***
# ***source json:tests/forte/data/ontology/test_specs/example_multi_module_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology example_multi_module_ontology. Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from typing import List
from typing import Optional


__all__ = [
    "Token",
    "Sentence",
    "Document",
]


class Token(Annotation):
    """
    Attributes:
        _lemma (Optional[str])
        _is_verb (Optional[bool])
        _num_chars (Optional[int])
        _score (Optional[float])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._lemma: Optional[str] = None
        self._is_verb: Optional[bool] = None
        self._num_chars: Optional[int] = None
        self._score: Optional[float] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['lemma'] = state.pop('_lemma')
        state['is_verb'] = state.pop('_is_verb')
        state['num_chars'] = state.pop('_num_chars')
        state['score'] = state.pop('_score')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._lemma = state.get('lemma', None) 
        self._is_verb = state.get('is_verb', None) 
        self._num_chars = state.get('num_chars', None) 
        self._score = state.get('score', None) 

    @property
    def lemma(self):
        return self._lemma

    @lemma.setter
    def lemma(self, lemma: Optional[str]):
        self.set_fields(_lemma=lemma)

    @property
    def is_verb(self):
        return self._is_verb

    @is_verb.setter
    def is_verb(self, is_verb: Optional[bool]):
        self.set_fields(_is_verb=is_verb)

    @property
    def num_chars(self):
        return self._num_chars

    @num_chars.setter
    def num_chars(self, num_chars: Optional[int]):
        self.set_fields(_num_chars=num_chars)

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score: Optional[float]):
        self.set_fields(_score=score)


class Sentence(Annotation):
    """
    Attributes:
        _tokens (Optional[List[int]])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._tokens: Optional[List[int]] = []

    def __getstate__(self): 
        state = super().__getstate__()
        state['tokens'] = state.pop('_tokens')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._tokens = state.get('tokens', None) 

    @property
    def tokens(self):
        return [self.pack.get_entry(tid) for tid in self._tokens]

    @tokens.setter
    def tokens(self, tokens: Optional[List[Token]]):
        tokens = [] if tokens is None else tokens
        self.set_fields(_tokens=[obj.tid for obj in tokens])

    def num_tokens(self):
        return len(self._tokens)

    def clear_tokens(self):
        [self.pack.delete_entry(self.pack.get_entry(tid)) for tid in self._tokens]
        self._tokens.clear()

    def add_tokens(self, a_tokens: Token):
        self._tokens.append(a_tokens.tid)


class Document(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
