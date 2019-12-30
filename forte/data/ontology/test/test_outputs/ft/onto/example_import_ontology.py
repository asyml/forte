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
    """
    Base parent token entry
    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._pos: typing.Optional[str] = None
        self._lemma: typing.Optional[str] = None

    @property
    def pos(self):
        return self._pos

    def set_pos(self, pos: typing.Optional[str]):
        self.set_fields(_pos=pos)

    @property
    def lemma(self):
        return self._lemma

    def set_lemma(self, lemma: typing.Optional[str]):
        self.set_fields(_lemma=lemma)


__all__.extend('EntityMention')


class EntityMention(forte.data.ontology.top.Annotation):

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._entity_type: typing.Optional[str] = None

    @property
    def entity_type(self):
        return self._entity_type

    def set_entity_type(self, entity_type: typing.Optional[str]):
        self.set_fields(_entity_type=entity_type)
