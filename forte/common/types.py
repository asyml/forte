from typing import Tuple, List, Dict, Type, Union

from forte.data.ontology.core import Entry, EntryType, GroupType, LinkType
from forte.data.base import Span

__all__ = [
    "EntryType",
    "GroupType",
    "LinkType",
    "ReplaceOperationsType",
    "DataRequest",
]

ReplaceOperationsType = List[Tuple[Span, str]]

DataRequest = Dict[Type[Entry], Union[Dict, List]]
