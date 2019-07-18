"""
The ontology used in our dummy relation extraction processor.

This is an example of multilevel inheritance:
    base_ontology->ontonotes_ontology->relation_ontology

This is also an example of extending the parent ontology by add a new entry.

"""
# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from nlp.pipeline.data.ontology.ontonotes_ontology import *


class RelationLink(Link):

    parent_type = "EntityMention"
    child_type = "EntityMention"

    def __init__(self, component: str):
        super().__init__(component)
        self.rel_type = None
