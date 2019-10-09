"""
Read wikipedia articles and parse each article into a DataPack, which contains
some basic structure of a Wikipedia article, such as links, categories and
info-boxes, see `forte.data.ontology.wiki_ontology` for more information.

This reader is based on DBpedia's extracted datasets.
"""
from typing import Any, Iterator, Dict, Type, Union, List

from forte.data import DataPack
from forte.data.ontology import Entry
from forte.data.ontology.wiki_ontology import WikiPage, WikiSection, \
    WikiParagraph, WikiTitle
from forte.data.readers import PackReader
from forte.data.datasets.wikipedia.db_utils import (
    NIFParser, NIFBufferedContextReader, get_resource_attribute,
    get_resource_name, get_resource_fragment, load_redirects
)


class DBpediaWikiReader(PackReader):
    def initialize(self, resource: Resources, configs: HParams):
        redirects = load_redirects(wiki_redirect)

    def define_output_info(self) -> Dict[Type[Entry], Union[List, Dict]]:
        pass

    def _collect(
            self, nif_context: str, nif_page_structure: str,
            mapping_literals: str, mapping_objects: str,
            nif_text_links: str, wiki_redirect: str,
    ) -> Iterator[Dict[str, List[str]]]:
        doc_data = {
            'doc_id': '',
            'context': '',
        }

        struct_reader = NIFBufferedContextReader(nif_page_structure)

        for context_statements in NIFParser(nif_context):
            for s, v, o, c in context_statements:
                nif_type = get_resource_attribute(s, "nif")

                if nif_type and nif_type == "context" and get_resource_fragment(
                        v) == 'isString':
                    doc_data['text'] = str(o)
                    doc_data['doc_name'] = get_resource_name(s)

                    doc_data['struct'] = struct_reader.get(c)

                    doc_data['oldid'] = get_resource_attribute(
                        c.identifier, 'oldid')

                    yield doc_data

    def parse_pack(self, doc_data: Dict[str, Any]) -> DataPack:
        pack = DataPack()
        full_text = doc_data['text']

        pack.set_text(full_text)
        page = WikiPage(pack, 0, len(full_text))
        pack.add_entry(page)
        page.set_page_id(doc_data['oldid'])

        for nif_range, rel, struct_type in doc_data['struct']:
            r = get_resource_fragment(rel)
            if r == 'type':
                range_ = get_resource_attribute(nif_range, 'char')
                begin, end = [int(d) for d in range_.split(',')]

                struct_ = get_resource_fragment(struct_type)

                if struct_ == 'Section':
                    section = WikiSection(pack, begin, end)
                    pack.add_entry(section)
                elif struct_ == 'Paragraph':
                    para = WikiParagraph(pack, begin, end)
                    pack.add_entry(para)
                elif struct_ == 'Title':
                    title = WikiTitle(pack, begin, end)
                    pack.add_entry(title)
                else:
                    print('raw struct')
                    print(struct_type)
                    print('fragment')
                    print(struct_)
                    input('new struct type?')

        pack.meta.doc_id = doc_data['doc_name']

        return pack

    def _cache_key_function(self, collection: Any) -> str:
        pass

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure
        :return:
        """
        return {
            'redirect_path': None,
        }
