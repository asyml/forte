"""
Process a Wikipedia dump and parse each article into a DataPack, which contains
some basic structure of a Wikipedia article, such as links, categories and
info-boxes, see `forte.data.ontology.wiki_ontology` for more information.
"""

import logging
import sys
from typing import NamedTuple
from io import StringIO
from typing import Any, Iterator, Dict, Iterable, Tuple, Set, Optional, List

import mwtypes
import mwxml
from forte.data.datasets.wikipedia.libs.common import Span
from forte.data.datasets.wikipedia.libs.wikilink import Wikilink

from forte.data import DataPack
from forte.data.ontology import wiki_ontology
from forte.data.ontology.wiki_ontology import WikiAnchor
from forte.data.readers.base_reader import PackReader
from forte.data.datasets.wikipedia.libs import wikilink_extractor
from forte.data.datasets.wikipedia.libs.WikiExtractor import Extractor

__all__ = [
    "WikiDumpReader",
]
logger = logging.getLogger(__name__)


class PendingAnnotation:
    def __init__(self, begin: int, end: int, text: str, anno_type: str):
        self.begin: int = begin
        self.end: int = end
        self.text: str = text
        self.anno_type: str = anno_type

    def __lt__(self, other):
        if self.begin == other.begin:
            return self.end < other.end
        return self.begin < other.begin

    def __eq__(self, other):
        return (self.begin, self.end) == (other.begin, other.end)


class WikiDumpReader(PackReader):

    def __init__(self, links_to_ignore: Optional[Set[str]] = None):
        super().__init__()
        self._ontology = wiki_ontology

        if links_to_ignore is None:
            # Default ignoring link types.
            self.links_to_ignore = {"File", "Category", "wikt"}
        else:
            self.links_to_ignore = links_to_ignore

        self.__pc = 0  # Process count.

    @property
    def pack_type(self):
        return DataPack

    def _cache_key_function(self, collection: Any) -> str:
        pass

    def define_output_info(self):
        # pylint: disable=no-self-use
        return {
            wiki_ontology.WikiPage: ["body"],
            wiki_ontology.WikiBody: ["introduction", "sections"],
            wiki_ontology.WikiSection: [],
            wiki_ontology.WikiAnchor: [],
            wiki_ontology.WikiAnchorLink: [],
            wiki_ontology.WikiInfoBox: ['text_entries', 'entity_entries'],
            wiki_ontology.WikiCategories: ['categories'],
        }

    def _collect(  # type: ignore
            self,
            wiki_dump_file: str,
            redirects: Optional[Dict[str, str]] = None,
    ) -> Iterator[Tuple]:
        """
        Create the wikipedia information tuples form the data.

        Args:
            wiki_dump_file: The original Wikipedia dump file
              (i.e. enwiki-yyyymmdd-pages-articles).
            redirects: A redirect dictionary. If provided, will resolve wiki
              page redirects from this dictionary.

        Returns:

        """

        if redirects is None:
            redirects = {}

        dump = mwxml.Dump.from_file(mwtypes.files.reader(wiki_dump_file))

        for (wiki_id, title, redirect, rev_id, wiki_links,
             text) in wikilink_extractor.main(dump, True):
            if redirect:
                # Ignoring redirect pages.
                continue

            title_norm = format_wiki_title(title)

            if title_norm in redirects:
                title_norm = redirects[title_norm]

            links: List[Tuple[str, str, Span]] = self.process_links(
                wiki_links, redirects)

            element = (wiki_id, title_norm, rev_id, links, text)
            yield element

    def parse_pack(self, collection: Tuple) -> DataPack:

        annotations: List[PendingAnnotation] = []

        pack = DataPack()

        wiki_id, title, rev_id, links, text = collection

        out = StringIO()
        Extractor(id, rev_id, title, text.split("\n")).extract(out)

        cleaned_text = out.getvalue()

        for anchor, link, span in links:
            annotations.append(PendingAnnotation(
                span.begin, span.end, anchor, 'WikiAnchor'
            ))

        anno_text = self.create_annotations(pack, text, annotations)

        print(
            f'length of origin {len(text)}, length of parsed {len(anno_text)}')
        input('check')

        self.set_text(pack, anno_text)
        pack.set_meta(doc_id=title)

        self.progress()

        return pack

    def create_annotations(
            self,
            pack: DataPack,
            text: str,
            annotations: List[PendingAnnotation]
    ) -> str:
        # annotations.sort(key=lambda a: a.span)

        for anno in sorted(annotations, reverse=True):
            print(anno.begin, anno.end)
            if anno.anno_type == 'WikiAnchor':
                anchor = WikiAnchor(pack, anno.begin, anno.end)
                text = text[:anno.begin] + anno.text + text[anno.end:]
                pack.add_entry(anchor)
        return text

    def progress(self):
        self.__pc += 1
        if self.__pc % 10 == 0:
            print('.', end='', file=sys.stderr, flush=True)
            if self.__pc % 500 == 0:
                print(self.__pc, file=sys.stderr, flush=True)

    def process_page_content(self):
        pass

    def process_links(
            self,
            wiki_links: Iterable[Tuple[Wikilink, Span]],
            redirects
    ) -> List[Tuple[str, str, Span]]:
        links: List[Tuple[str, str, Span]] = []
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


def format_wiki_title(link_target):
    """
    Normalize the link name of the link, such as replacing space, and first
    letter capitalization. See: https://en.wikipedia.org/wiki/Wikipedia:
    Naming_conventions_(capitalization)#Software_characteristics
    Args:
        link_target: The wiki link to be normalized.

    Returns:

    """
    return link_target.replace(" ", "_").capitalize()
