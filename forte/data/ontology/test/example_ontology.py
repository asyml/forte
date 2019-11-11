# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated file. Do not change manually.
"""
import typing
import forte.data.data_pack
import forte.data.ontology.top
import ft.onto.example_import_ontology


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
        self._string_features = string_features

    @property
    def word_forms(self):
        return self._word_forms

    def set_word_forms(self, word_forms: typing.Optional[typing.List[ft.onto.example_ontology.Word]]):
        self._word_forms = word_forms
