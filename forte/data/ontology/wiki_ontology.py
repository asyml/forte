from typing import List, Dict

import forte.data.ontology.base_ontology as ontology
from forte.data.ontology.top import Entry, Annotation, Link, Group
from forte.data.ontology.top_multipack import MultiPackLink


class WikiPage(ontology.Document):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.body: WikiBody


class WikiBody(Annotation):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.introduction: WikiSection
        self.sections: List[WikiSection]


class WikiSection(Annotation):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)


class WikiAnchor(Annotation):
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)


class WikiAnchorLink(MultiPackLink):
    def __init__(self, anchor: WikiAnchor, page: WikiPage):
        super().__init__(anchor, page)


class WikiInfoBox(Entry):
    def __init__(self):
        super().__init__()
        self.text_entries: Dict[str, str] = {}
        self.entity_entries: Dict[str, WikiPage] = {}


class WikiCategories(Entry):
    def __init__(self):
        super().__init__()
        self.categories: List[str] = []
