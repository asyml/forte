from typing import Tuple, List

from forte.data.ontology.core import EntryType, GroupType, LinkType

__all__ = [
    "EntryType",
    "GroupType",
    "LinkType",
    "ReplaceOperationsType",
]

ReplaceOperationsType = List[Tuple[Tuple[int, int], str]]
