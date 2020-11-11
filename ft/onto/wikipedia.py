# ***automatically_generated***
# ***source json:forte/ontology_specs/wikipedia.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology wikipedia. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Generics
from typing import Optional

__all__ = [
    "WikiPage",
    "WikiBody",
    "WikiSection",
    "WikiParagraph",
    "WikiTitle",
    "WikiAnchor",
    "WikiInfoBoxProperty",
    "WikiInfoBoxMapped",
]


@dataclass
class WikiPage(Annotation):
    """
    Attributes:
        page_id (Optional[str])
        page_name (Optional[str])
    """

    page_id: Optional[str]
    page_name: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.page_id: Optional[str] = None
        self.page_name: Optional[str] = None


@dataclass
class WikiBody(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class WikiSection(Annotation):
    """
    Attributes:
        is_intro (Optional[bool])
    """

    is_intro: Optional[bool]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.is_intro: Optional[bool] = None


@dataclass
class WikiParagraph(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class WikiTitle(Annotation):

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class WikiAnchor(Annotation):
    """
    Attributes:
        target_page_name (Optional[str])
    """

    target_page_name: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.target_page_name: Optional[str] = None


@dataclass
class WikiInfoBoxProperty(Generics):
    """
    Attributes:
        key (Optional[str])
        value (Optional[str])
    """

    key: Optional[str]
    value: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.key: Optional[str] = None
        self.value: Optional[str] = None


@dataclass
class WikiInfoBoxMapped(Generics):
    """
    Attributes:
        key (Optional[str])
        value (Optional[str])
    """

    key: Optional[str]
    value: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.key: Optional[str] = None
        self.value: Optional[str] = None
