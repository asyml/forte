from typing import Tuple, List

from forte.data.ontology.core import EntryType, GroupType, LinkType
from forte.data.base import Span

__all__ = [
    "EntryType",
    "GroupType",
    "LinkType",
    "ReplaceOperationsType",
]

ReplaceOperationsType = List[Tuple[Span, str]]
