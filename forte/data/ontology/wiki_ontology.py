from typing import List, Dict

import ft.onto.base_ontology as base_ontology
from forte.data.data_pack import DataPack
from forte.data.ontology.base.top import Entry, Annotation, Link


class WikiPage(base_ontology.Document):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.body: WikiBody


class WikiBody(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.introduction: WikiSection
        self.sections: List[WikiSection]


class WikiSection(Annotation):
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
        self.text_entries: Dict[str, str] = {}
        self.entity_entries: Dict[str, WikiPage] = {}


class WikiCategories(Entry):
    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.categories: List[str] = []
