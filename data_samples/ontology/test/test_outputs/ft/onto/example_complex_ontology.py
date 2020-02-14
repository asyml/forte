# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""


Automatically generated ontology . Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Link
from typing import List
from typing import Optional


__all__ = [
    "Token",
    "Sentence",
    "Document",
    "Dependency",
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
        key_tokens (Optional[List[Token]])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.key_tokens: Optional[List[int]] = []

    def __getstate__(self): 
        state = super().__getstate__()
        state['key_tokens'] = self.key_tokens
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.key_tokens = state.get('key_tokens', None) 

    @property
    def key_tokens(self):
        return self.key_tokens

    @key_tokens.setter
    def key_tokens(self, key_tokens: Optional[List[Token]]):
        self.set_fields(key_tokens=[self.__pack.add_entry_(obj) for obj in key_tokens])

    def num_key_tokens(self):
        return len(self.key_tokens)

    def clear_key_tokens(self):
        [self.__pack.delete_entry(self.__pack.get_entry(tid)) for tid in self.key_tokens]
        self.key_tokens.clear()

    def add_key_tokens(self, a_key_tokens: Token):
        self.key_tokens.append(self.__pack.add_entry_(a_key_tokens))


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


class Dependency(Link):
    """

    Attributes:
        rel_type (Optional[str])

    """

    def __init__(self, pack: DataPack, parent: Optional[DataPack] = None, child: Optional[DataPack] = None):
        super().__init__(pack, parent, child)
        self.rel_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['rel_type'] = self.rel_type
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.rel_type = state.get('rel_type', None) 

    @property
    def rel_type(self):
        return self.rel_type

    @rel_type.setter
    def rel_type(self, rel_type: Optional[str]):
        self.set_fields(rel_type=rel_type)
