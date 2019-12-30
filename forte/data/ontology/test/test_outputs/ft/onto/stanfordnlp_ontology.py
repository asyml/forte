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
        self._pos_tag: typing.Optional[str] = None
        self._upos: typing.Optional[str] = None
        self._xpos: typing.Optional[str] = None

    @property
    def lemma(self):
        return self._lemma

    def set_lemma(self, lemma: typing.Optional[str]):
        self.set_fields(_lemma=lemma)

    @property
    def pos_tag(self):
        return self._pos_tag

    def set_pos_tag(self, pos_tag: typing.Optional[str]):
        self.set_fields(_pos_tag=pos_tag)

    @property
    def upos(self):
        return self._upos

    def set_upos(self, upos: typing.Optional[str]):
        self.set_fields(_upos=upos)

    @property
    def xpos(self):
        return self._xpos

    def set_xpos(self, xpos: typing.Optional[str]):
        self.set_fields(_xpos=xpos)


__all__.extend('Sentence')


class Sentence(forte.data.ontology.top.Annotation):

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._tokens: typing.Optional[typing.List[ft.onto.stanfordnlp_ontology.Token]] = None

    @property
    def tokens(self):
        return self._tokens

    def set_tokens(self, tokens: typing.Optional[typing.List[ft.onto.stanfordnlp_ontology.Token]]):
        self.set_fields(_tokens=[item.tid for item in tokens])


__all__.extend('Document')


class Document(forte.data.ontology.top.Annotation):

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)


__all__.extend('Dependency')


class Dependency(forte.data.ontology.top.Link):
    parent_type: ft.onto.stanfordnlp_ontology.Token = None
    child_type: ft.onto.stanfordnlp_ontology.Token = None

    def __init__(self, pack: forte.data.base_pack.PackType, parent: typing.Optional[forte.data.ontology.core.Entry] = None, child: typing.Optional[forte.data.ontology.core.Entry] = None):
        super().__init__(pack, parent, child)
        self._rel_type: typing.Optional[str] = None

    @property
    def rel_type(self):
        return self._rel_type

    def set_rel_type(self, rel_type: typing.Optional[str]):
        self.set_fields(_rel_type=rel_type)
