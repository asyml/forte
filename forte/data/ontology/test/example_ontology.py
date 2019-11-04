# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: disable
"""
Automatically generated file. Do not change manually.
"""
import typing
import forte.data.data_pack
import forte.data.ontology.top
import ft.onto.example_import_ontology


__all__ = []


__all__.extend('Token')


class Token(ft.onto.example_import_ontology.Token):
    """
        Args:
            related_tokens (typing.Optional[typing.List[str]]): Tokens related to the current token
            string_features (typing.Optional[typing.List["Token"]]): Miscellaneous string features
    """
    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._related_tokens: typing.Optional[typing.List[str]] = None
        self._string_features: typing.Optional[typing.List["Token"]] = None

    @property
    def related_tokens(self):
        return self._related_tokens

    def set_related_tokens(self, related_tokens: typing.Optional[typing.List[str]]):
        self._related_tokens = related_tokens

    @property
    def string_features(self):
        return self._string_features

    def set_string_features(self, string_features: typing.Optional[typing.List["Token"]]):
        self._string_features = string_features
