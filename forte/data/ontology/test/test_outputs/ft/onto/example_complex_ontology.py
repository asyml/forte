# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated file. Do not change manually.
"""
import forte.data.data_pack
import forte.data.ontology.top
import ft.onto
import typing


__all__ = []


__all__.extend('Token')


class Token(forte.data.ontology.top.Annotation):

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._lemma: typing.Optional[str] = None
        self._is_verb: typing.Optional[bool] = None
        self._num_chars: typing.Optional[int] = None
        self._score: typing.Optional[float] = None

    @property
    def lemma(self):
        return self._lemma

    def set_lemma(self, lemma: typing.Optional[str]):
        self.set_fields(_lemma=lemma)

    @property
    def is_verb(self):
        return self._is_verb

    def set_is_verb(self, is_verb: typing.Optional[bool]):
        self.set_fields(_is_verb=is_verb)

    @property
    def num_chars(self):
        return self._num_chars

    def set_num_chars(self, num_chars: typing.Optional[int]):
        self.set_fields(_num_chars=num_chars)

    @property
    def score(self):
        return self._score

    def set_score(self, score: typing.Optional[float]):
        self.set_fields(_score=score)


__all__.extend('Sentence')


class Sentence(forte.data.ontology.top.Annotation):

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._tokens: typing.Optional[typing.List[ft.onto.example_complex_ontology.Token]] = None

    @property
    def tokens(self):
        return self._tokens

    def set_tokens(self, tokens: typing.Optional[typing.List[ft.onto.example_complex_ontology.Token]]):
        self.set_fields(_tokens=[item.tid for item in tokens])


__all__.extend('Document')


class Document(forte.data.ontology.top.Annotation):

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)


__all__.extend('Dependency')


class Dependency(forte.data.ontology.top.Link):

    def __init__(self, pack: forte.data.base_pack.PackType, parent: typing.Optional[forte.data.ontology.core.Entry] = None, child: typing.Optional[forte.data.ontology.core.Entry] = None):
        super().__init__(pack, parent, child)
        self._rel_type: typing.Optional[str] = None

    @property
    def rel_type(self):
        return self._rel_type

    def set_rel_type(self, rel_type: typing.Optional[str]):
        self.set_fields(_rel_type=rel_type)
