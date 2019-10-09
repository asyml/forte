from typing import List, Dict

import forte.data.ontology.base_ontology as ontology
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation, Link
from forte.data.ontology.core import Entry


class WikiPage(ontology.Document):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._body: WikiBody
        self._page_id: str

    def set_page_id(self, pid: str):
        self.set_fields(_page_id=pid)

    @property
    def page_id(self):
        return self._page_id


class WikiBody(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class WikiSection(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._is_intro: bool

    def set_is_intro(self, is_intro: bool):
        self.set_is_intro(is_intro)

    @property
    def is_intro(self):
        return self._is_intro


class WikiParagraph(Annotation):
    pass


class WikiTitle(Annotation):
    pass


class WikiAnchor(Annotation):
    pass


class WikiAnchorLink(Link):
    ParentType = WikiAnchor
    ChildType = WikiPage

    def __init__(self, pack: DataPack, anchor: WikiAnchor, page: WikiPage):
        super().__init__(pack, anchor, page)


class WikiInfoBox(Entry):
    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self._literal_entries: Dict[str, str] = {}
        self._object_entries: Dict[str, str] = {}


class WikiCategories(Entry):
    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self._categories: List[str] = []
