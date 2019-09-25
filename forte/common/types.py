from typing import TypeVar, Tuple, List

from forte.data.ontology.top import Entry, BaseGroup, BaseLink

__all__ = [
    "EntryType",
    "GroupType",
    "LinkType",
    "ReplaceOperationsType",
]

EntryType = TypeVar("EntryType", bound=Entry)
GroupType = TypeVar("GroupType", bound=BaseGroup)
LinkType = TypeVar('LinkType', bound=BaseLink)

ReplaceOperationsType = List[Tuple[Tuple[int, int], str]]
