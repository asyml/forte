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
import ft.onto.example_import_ontology
import typing


__all__ = []


__all__.extend('Word')


class Word(ft.onto.example_import_ontology.Token):
    """
        Args:
            string_features (typing.Optional[typing.List[str]]): To demonstrate the composite type, List.
            word_forms (typing.Optional[typing.List[ft.onto.example_ontology.Word]]): To demonstrate that an attribute can have the entry it is contained in as it's type or item_type.
    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._string_features: typing.Optional[typing.List[str]] = None
        self._word_forms: typing.Optional[typing.List[ft.onto.example_ontology.Word]] = None

    @property
    def string_features(self):
        return self._string_features

    def set_string_features(self, string_features: typing.Optional[typing.List[str]]):
        self.set_fields(_string_features=[item for item in string_features])

    @property
    def word_forms(self):
        return self._word_forms

    def set_word_forms(self, word_forms: typing.Optional[typing.List[ft.onto.example_ontology.Word]]):
        self.set_fields(_word_forms=[item.tid for item in word_forms])


__all__.extend('WordLink')


class WordLink(forte.data.ontology.top.Link):
    """
        Args:
            string_features (typing.Optional[typing.List[str]]): To demonstrate the composite type, List.
    """
    parent_type: ft.onto.example_ontology.Word = None
    child_type: ft.onto.example_ontology.Word = None

    def __init__(self, pack: forte.data.base_pack.PackType, parent: typing.Optional[forte.data.ontology.core.Entry] = None, child: typing.Optional[forte.data.ontology.core.Entry] = None):
        super().__init__(pack, parent, child)
        self._string_features: typing.Optional[typing.List[str]] = None

    @property
    def string_features(self):
        return self._string_features

    def set_string_features(self, string_features: typing.Optional[typing.List[str]]):
        self.set_fields(_string_features=[item for item in string_features])
