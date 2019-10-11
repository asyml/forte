# flake8: noqa
# mypy: ignore-errors
# pylint: disable=line-too-long,trailing-newlines
"""
Automatically generated file. Do not change by hand
"""

import typing
import forte.data.data_pack
import forte.data.ontology.base.top
import forte.data.ontology.example_import_ontology


__all__ = []

__all__.extend("Token")


class Token(forte.data.ontology.example_import_ontology.Token):
    def __init__(self, pack: forte.data.container.EntryContainer, begin: int, end: int):
        """
        Attributes:
            related_tokens (typing.List): Tokens related to the current token
            string_features (typing.List): Miscellaneous string features
        """
        super().__init__(pack, begin, end)
        self._related_tokens: typing.Optional[typing.List[forte.data.ontology.example_ontology.Token]] = None
        self._string_features: typing.Optional[typing.List[str]] = None

    @property
    def related_tokens(self):
        return self._related_tokens

    def set_related_tokens(self, related_tokens: typing.List):
        self.set_fields(_related_tokens=related_tokens)

    @property
    def string_features(self):
        return self._string_features

    def set_string_features(self, string_features: typing.List):
        self.set_fields(_string_features=string_features)


