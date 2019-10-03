# mypy: ignore-errors
"""
Ontology file for forte.data.ontology.example_ontology
Automatically generated file. Do not change by hand
"""
import typing
import forte.data.data_pack
import forte.data.ontology.top


class Token(forte.data.ontology.top.Annotation):
    def __init__(self, pack:forte.data.data_pack.DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._pos_tag: typing.Optional[str] = None
        self._lemma: typing.Optional[str] = None
        self._related_tokens: typing.Optional[typing.List[forte.data.ontology.example_ontology.Token]] = None
        self._string_features: typing.Optional[typing.List[str]] = None

    @property
    def pos_tag(self):
        return self._pos_tag

    def set_pos_tag(self, pos_tag: str):
        self.set_fields(_pos_tag=pos_tag)

    @property
    def lemma(self):
        return self._lemma

    def set_lemma(self, lemma: str):
        self.set_fields(_lemma=lemma)

    @property
    def related_tokens(self):
        return self._related_tokens

    def set_related_tokens(self, related_tokens: List):
        self.set_fields(_related_tokens=related_tokens)

    @property
    def string_features(self):
        return self._string_features

    def set_string_features(self, string_features: List):
        self.set_fields(_string_features=string_features)


class EntityMention(forte.data.ontology.top.Annotation):
    def __init__(self, pack:forte.data.data_pack.DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._entity_type: typing.Optional[str] = None

    @property
    def entity_type(self):
        return self._entity_type

    def set_entity_type(self, entity_type: str):
        self.set_fields(_entity_type=entity_type)
