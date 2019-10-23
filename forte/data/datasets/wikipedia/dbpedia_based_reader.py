"""
Read wikipedia articles and parse each article into a DataPack, which contains
some basic structure of a Wikipedia article, such as links, categories and
info-boxes, see `forte.data.ontology.wiki_ontology` for more information.

This reader is based on DBpedia's extracted datasets.
"""
import logging
from collections import defaultdict
from typing import Any, Iterator, Dict, Type, Union, List, DefaultDict, Tuple

from texar.torch import HParams
import rdflib

from forte import Resources
from forte.data import DataPack
from forte.data.datasets.wikipedia.db_utils import (
    NIFParser, NIFBufferedContextReader, get_resource_attribute,
    get_resource_name, get_resource_fragment,
    print_progress)
from forte.data.ontology import Entry
from forte.data.readers import PackReader
from ft.onto.wikipedia import (WikiPage, WikiSection, WikiParagraph, WikiTitle,
                               WikiAnchor, WikiInfoBoxMapped)

state_type = Tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]


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
                logging.warning("Unknown struct type: %s", struct_type)


def add_anchor_links(pack: DataPack, text_link_statements: List[state_type],
                     redirects: Dict[str, str]):
    link_grouped: DefaultDict[str,
                              Dict[str, rdflib.term.Node]] = defaultdict(dict)
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
                    logging.warning("Unknown anchor type: %s", info_value)
            if info_key == 'taIdentRef':
                target_page_name = get_resource_name(info_value)
                if target_page_name in redirects:
                    target_page_name = redirects[target_page_name]
                anchor.set_target_page_name(target_page_name)
        pack.add_entry(anchor)


def add_info_boxes(pack: DataPack, info_box_statements: List):
    for _, v, o in info_box_statements:
        slot_name = v.toPython()
        slot_value = get_resource_name(o)
        info_box = WikiInfoBoxMapped(pack)
        info_box.set_key(slot_name)
        info_box.set_value(slot_value)
        pack.add_entry(info_box)


class DBpediaWikiReader(PackReader):
    def __init__(self):
        super().__init__()
        self.struct_reader = None
        self.link_reader = None
        self.redirects: Dict[str, str] = {}

    def initialize(self, resource: Resources, configs: HParams):
        self.redirects = resource.get('redirects')

        # These NIF readers organize the statements in the specific RDF context,
        # in this case each context correspond to one wiki page, this allows
        # us to read the information more systematically.
        self.struct_reader = NIFBufferedContextReader(
            configs.nif_page_structure)
        self.link_reader = NIFBufferedContextReader(configs.nif_text_links)

    def define_output_info(self) -> Dict[Type[Entry], Union[List, Dict]]:
        pass

    def _collect(self, nif_context: str  # type: ignore
                 ) -> Iterator[Tuple[Dict[str, str],
                                     Dict[str, List[state_type]]]]:
        str_data: Dict[str, str] = {}
        node_data: Dict[str, List[state_type]] = {}

        for context_statements in NIFParser(nif_context):
            for s, v, o, c in context_statements:
                nif_type = get_resource_attribute(s, "nif")
                print_progress(f'Collecting DBpedia context: [{c.identifier}]')

                if nif_type and nif_type == "context" and get_resource_fragment(
                        v) == 'isString':
                    str_data['text'] = o.toPython()
                    str_data['doc_name'] = get_resource_name(s)
                    str_data['oldid'] = get_resource_attribute(
                        c.identifier, 'oldid')

                    node_data['struct'] = self.struct_reader.get(c)
                    node_data['links'] = self.link_reader.get(c)

                    yield str_data, node_data
        print(' ..Done')

    def _parse_pack(
            self, doc_data: Tuple[Dict[str, str], Dict[str, List[state_type]]]
    ) -> Iterator[DataPack]:
        str_data, node_data = doc_data

        pack = DataPack()
        doc_name: str = str_data['doc_name']
        if doc_name in self.redirects:
            doc_name = self.redirects[doc_name]

        full_text: str = str_data['text']

        pack.set_text(full_text)
        page = WikiPage(pack, 0, len(full_text))
        pack.add_entry(page)
        page.set_page_id(str_data['oldid'])
        page.set_page_name(doc_name)

        if len(node_data['struct']) > 0:
            add_struct(pack, node_data['struct'])
        else:
            logging.warning('Structure info for %s not found.', doc_name)

        if len(node_data['links']) > 0:
            add_anchor_links(pack, node_data['links'], self.redirects)
        else:
            logging.warning('Links for [%s] not found.', doc_name)

        pack.meta.doc_id = doc_name

        yield pack

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
            'nif_page_structure': None,
            'nif_text_links': None,
        }
