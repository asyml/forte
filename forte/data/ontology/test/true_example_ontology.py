"""
Automatically generated file. Do not change by hand.
"""
import typing
import forte.data.ontology
import forte.data.ontology.top


class Token(forte.data.ontology.top.Annotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    pos_tag: typing.Optional[str] = None
    lemma: typing.Optional[str] = None


class EntityMention(forte.data.ontology.top.Annotation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    entity_type: typing.Optional[str] = None
