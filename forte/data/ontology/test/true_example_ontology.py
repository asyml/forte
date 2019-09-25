"""
Ontology file for forte.data.ontology.example_ontology
Automatically generated file. Do not change by hand
"""
import typing
import forte.data.ontology.top


class Token(forte.data.ontology.top.Annotation):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self._pos_tag: typing.Optional[str] = None
        self._lemma: typing.Optional[str] = None

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


class EntityMention(forte.data.ontology.top.Annotation):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self._entity_type: typing.Optional[str] = None

    @property
    def entity_type(self):
        return self._entity_type

    def set_entity_type(self, entity_type: str):
        self.set_fields(_entity_type=entity_type)
