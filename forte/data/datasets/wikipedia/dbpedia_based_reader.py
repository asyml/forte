"""
Read wikipedia articles and parse each article into a DataPack, which contains
some basic structure of a Wikipedia article, such as links, categories and
info-boxes, see `forte.data.ontology.wiki_ontology` for more information.

This reader is based on DBpedia's extracted datasets.
"""
import logging
from collections import defaultdict
from typing import Any, Iterator, Dict, Type, Union, List

from texar.torch import HParams

from forte import Resources
from forte.data import DataPack
from forte.data.ontology import Entry
from forte.data.ontology.wiki_ontology import WikiPage, WikiSection, \
    WikiParagraph, WikiTitle, WikiAnchor, WikiInfoBoxEntry
from forte.data.readers import PackReader
from forte.data.datasets.wikipedia.db_utils import (
    NIFParser, NIFBufferedContextReader, get_resource_attribute,
    get_resource_name, get_resource_fragment, load_redirects
)


def add_struct(pack: DataPack, struct_statements: List):
    for nif_range, rel, struct_type in struct_statements:
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


def add_anchor_links(pack: DataPack, text_link_statements: List):
    link_grouped = defaultdict(dict)
    for nif_range, rel, info in text_link_statements:
        range_ = get_resource_attribute(nif_range, 'char')
        r = get_resource_fragment(rel)
        link_grouped[range_][r] = info

    for range_, link_infos in link_grouped.items():
        begin, end = [int(d) for d in range_.split(',')]
        anchor = WikiAnchor(pack, begin, end)
        for info_key, info_value in link_infos.items():
            if info_key == 'type':
                anchor_type = get_resource_fragment(info_value)
                if not anchor_type == 'Phrase' and not anchor_type == 'Word':
                    print(info_value)
                    input('unknown anchor type.')
            if info_key == 'taIdentRef':
                anchor.set_target_page_name(get_resource_name(info_value))
        pack.add_entry(anchor)


def add_info_boxes(pack: DataPack, info_box_statements: List):
    for _, v, o in info_box_statements:
        slot_name = v.toPython()
        slot_value = get_resource_name(o)
        info_box = WikiInfoBoxEntry(pack)
        info_box.set_key(slot_name)
        info_box.set_value(slot_value)
        pack.add_entry(info_box)


class DBpediaWikiReader(PackReader):
    def initialize(self, resource: Resources, configs: HParams):
        # self.redirects = load_redirects(configs.redirect_path)
        pass

    def define_output_info(self) -> Dict[Type[Entry], Union[List, Dict]]:
        pass

    def _collect(
            self, nif_context: str, nif_page_structure: str,
            mapping_literals: str, mapping_objects: str,
            nif_text_links: str
    ) -> Iterator[Dict[str, List[str]]]:
        doc_data = {
            'doc_id': '',
            'context': '',
        }

        # These NIF readers organize the statements in the specific RDF context,
        # in this case each context correspond to one wiki page, this allows
        # us to read the information more systematically.
        struct_reader = NIFBufferedContextReader(nif_page_structure)
        link_reader = NIFBufferedContextReader(nif_text_links)
        literal_info_reader = NIFBufferedContextReader(mapping_literals)
        object_info_reader = NIFBufferedContextReader(mapping_objects)

        for context_statements in NIFParser(nif_context):
            for s, v, o, c in context_statements:
                nif_type = get_resource_attribute(s, "nif")

                if nif_type and nif_type == "context" and get_resource_fragment(
                        v) == 'isString':
                    doc_data['text'] = o.toPython()
                    doc_data['doc_name'] = get_resource_name(s)
                    doc_data['struct'] = struct_reader.get(c)
                    doc_data['links'] = link_reader.get(c)
                    doc_data['literal_info'] = literal_info_reader.get(c)
                    doc_data['object_info'] = object_info_reader.get(c)
                    doc_data['oldid'] = get_resource_attribute(
                        c.identifier, 'oldid')

                    yield doc_data

    def parse_pack(self, doc_data: Dict[str, Any]) -> DataPack:
        pack = DataPack()
        doc_name = doc_data['doc_name']

        full_text = doc_data['text']

        pack.set_text(full_text)
        page = WikiPage(pack, 0, len(full_text))
        pack.add_entry(page)
        page.set_page_id(doc_data['oldid'])
        page.set_page_name(doc_name)

        if len(doc_data['struct']) == 0:
            logging.warning(f'Structure info for [{doc_name}] not found.')
        else:
            add_struct(pack, doc_data['struct'])

        if len(doc_data['links']) == 0:
            logging.warning(f'Links for [{doc_name}] not found.')
        else:
            add_anchor_links(pack, doc_data['links'])

        if len(doc_data['literal_info']) == 0:
            logging.warning(f'Literal info boxes for [{doc_name}] not found.')
        else:
            add_info_boxes(pack, doc_data['literal_info'])

        if len(doc_data['object_info']) == 0:
            logging.warning(f'Object info boxes for [{doc_name}] not found.')
        else:
            add_info_boxes(pack, doc_data['object_info'])

        pack.meta.doc_id = doc_name

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
