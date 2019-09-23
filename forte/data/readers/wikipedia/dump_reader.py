"""
Process a Wikipedia dump and parse each article into a DataPack, which contains
some basic structure of a Wikipedia article, such as links, categories and
info-boxes, see `forte.data.ontology.wiki_ontology` for more information.

This reader depends on python-mwlinks [1], To use this reader, you have to add
the utilities from MediaWiki into your PYTHONPATH.

[1] https://github.com/mediawiki-utilities/python-mwlinks
"""

import os
import logging
from typing import Any, Iterator, Dict, Iterable, Tuple, Set

import mwxml
import mwtypes
from mwlinks.libs.wikilink import Wikilink
from mwlinks.libs.common import Span
from mwlinks.libs.WikiExtractor import Extractor
from multiprocessing import Pool, Value, Lock, Queue, Manager

from forte.data.readers.wikipedia import page_parser
from forte.data import DataPack
from forte.data.readers.base_reader import PackReader
from forte.data.ontology import wiki_ontology

__all__ = [
    "WikiDumpReader",
]
logger = logging.getLogger(__name__)


class WikiDumpReader(PackReader):
    def __init__(self, links_to_ignore: Set[str] = None):
        super().__init__()
        self._ontology = wiki_ontology

        if links_to_ignore is None:
            # Default ignoring link types.
            self.links_to_ignore = {"File", "Category", "wikt"}
        else:
            self.links_to_ignore = links_to_ignore

    @property
    def pack_type(self):
        return DataPack

    def _cache_key_function(self, collection: Any) -> str:
        pass

    def define_output_info(self):
        self.output_info = {
            wiki_ontology.WikiPage: ["body"],
            wiki_ontology.WikiBody: ["introduction", "sections"],
            wiki_ontology.WikiSection: [],
            wiki_ontology.WikiAnchor: [],
            wiki_ontology.WikiAnchorLink: [],
            wiki_ontology.WikiInfoBox: ['text_entries', 'entity_entries'],
            wiki_ontology.WikiCategories: ['categories'],
        }

    def _collect(
            self,
            wiki_dump_file: str,
            redirects: Dict[str, str] = None,
    ) -> Iterator[Tuple]:
        if redirects is None:
            redirects = {}

        dump = mwxml.Dump.from_file(mwtypes.files.reader(wiki_dump_file))

        for (wiki_id, title, redirect, revision_id, wiki_links,
             text) in page_parser.parse(dump, True):
            if redirect:
                # Ignoring redirect pages.
                continue

            title_wiki_name = format_wiki_title(title)
            if title_wiki_name in redirects:
                title_wiki_name = redirects[title_wiki_name]

            links = self.process_links(wiki_links, redirects)

            print('next reading: ', wiki_id, title, 'contains', len(wiki_links))

            element = (wiki_id, title, redirect, revision_id, links, text)
            yield element

    def parse_pack(self, collection: Tuple) -> DataPack:
        pack = DataPack()

        wiki_id, title, redirect, revision_id, links, text = collection

        self.set_text(pack, text)
        pack.set_meta(doc_id=title)

        return pack

    def process_page_content(self):
        pass

    def process_links(self, wiki_links: Iterable[Tuple[Wikilink, Span]],
                      redirects):
        links = []
        for link, span in wiki_links:
            if not self.is_ignore_link(link):
                wiki_title = get_wiki_title(link.link, redirects)
                links.append((link.anchor, wiki_title, span))
        return links

    def is_ignore_link(self, link: Wikilink):
        link_parts = link.link.split(":")
        if len(link_parts) > 1:
            link_type = link_parts[0]
            if link_type in self.links_to_ignore:
                return True
        return False


def get_wiki_title(title, redirects):
    wiki_title = format_wiki_title(title)
    if wiki_title in redirects:
        wiki_title = redirects[wiki_title]
    return wiki_title


def format_wiki_title(link):
    """
    Normalize the link name of the link, such as replacing space, and first
    letter capitalization. See: https://en.wikipedia.org/wiki/Wikipedia:
    Naming_conventions_(capitalization)#Software_characteristics
    :param link:
    :return:
    """
    return link.replace(" ", "_").capitalize()
