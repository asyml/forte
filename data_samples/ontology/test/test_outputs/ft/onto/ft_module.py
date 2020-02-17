# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""


Automatically generated ontology . Do not change manually.
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
        lemma (Optional[str])
        is_verb (Optional[bool])
        num_chars (Optional[int])
        score (Optional[float])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.lemma: Optional[str] = None
        self.is_verb: Optional[bool] = None
        self.num_chars: Optional[int] = None
        self.score: Optional[float] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['lemma'] = self.lemma
        state['is_verb'] = self.is_verb
        state['num_chars'] = self.num_chars
        state['score'] = self.score
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.lemma = state.get('lemma', None) 
        self.is_verb = state.get('is_verb', None) 
        self.num_chars = state.get('num_chars', None) 
        self.score = state.get('score', None) 

    @property
    def lemma(self):
        return self.lemma

    @lemma.setter
    def lemma(self, lemma: Optional[str]):
        self.set_fields(lemma=lemma)

    @property
    def is_verb(self):
        return self.is_verb

    @is_verb.setter
    def is_verb(self, is_verb: Optional[bool]):
        self.set_fields(is_verb=is_verb)

    @property
    def num_chars(self):
        return self.num_chars

    @num_chars.setter
    def num_chars(self, num_chars: Optional[int]):
        self.set_fields(num_chars=num_chars)

    @property
    def score(self):
        return self.score

    @score.setter
    def score(self, score: Optional[float]):
        self.set_fields(score=score)


class Sentence(Annotation):
    """

    Attributes:
        tokens (Optional[List[int]])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.tokens: Optional[List[int]] = []

    def __getstate__(self): 
        state = super().__getstate__()
        state['tokens'] = self.tokens
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.tokens = state.get('tokens', []) 

    @property
    def tokens(self):
        return self.tokens

    @tokens.setter
    def tokens(self, tokens: Optional[List[Token]]):
        self.set_fields(tokens=[self.__pack.add_entry_(obj) for obj in tokens])

    def num_tokens(self):
        return len(self.tokens)

    def clear_tokens(self):
        [self.__pack.delete_entry(self.__pack.get_entry(tid)) for tid in self.tokens]
        self.tokens.clear()

    def add_tokens(self, a_tokens: Token):
        self.tokens.append(self.__pack.add_entry_(a_tokens))


class Document(Annotation):
    """


    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)

    def __getstate__(self): 
        state = super().__getstate__()
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
