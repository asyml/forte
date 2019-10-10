"""
The ontology used in our dummy relation extraction processor.

This is an example of multilevel inheritance:
    base_ontology->ontonotes_ontology->relation_ontology

This is also an example of extending the parent ontology by add a new entry.
"""
# pylint: disable=unused-wildcard-import, wildcard-import, function-redefined
from typing import Optional

from forte.data.ontology.ontonotes_ontology import *
from forte.data.ontology.top import Link


class RelationLink(Link):
    """
    A :class:`~forte.data.ontology.top.Link` type entry which take
    :class:`~forte.data.ontology.base_ontology.EntityMention`
    objects as parent and child.

    Args:
        pack (DataPack): the containing pack of this link.
        parent (Entry, optional): the parent entry of the link.
        child (Entry, optional): the child entry of the link.
    """
    ParentType = EntityMention
    """The entry type of the parent node of :class:`RelationLink`."""
    ChildType = EntityMention
    """The entry type of the child node of :class:`RelationLink`."""

    def __init__(
            self,
            pack: DataPack,
            parent: Optional[EntityMention] = None,
            child: Optional[EntityMention] = None):
        super().__init__(pack, parent, child)
        self.rel_type = None
