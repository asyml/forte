"""
The ontology used in our dummy relation extraction processor.

This is an example of extending the parent ontology by add a new entry.
"""
# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from typing import Optional
from nlp.pipeline.data.ontology.base_ontology import *
from nlp.pipeline.data.ontology.top import Link


class RelationLink(Link):
    """
    A :class:`Link` type entry which take :class:`EntityMention` objects as
    parent and child.
    """
    parent_type = EntityMention
    child_type = EntityMention

    def __init__(self, parent: Optional[EntityMention] = None,
                 child: Optional[EntityMention] = None):
        super().__init__(parent, child)
        self.rel_type = None
