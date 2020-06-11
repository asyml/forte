# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Read wikipedia articles and parse each article into a DataPack, which contains
some basic structure of a Wikipedia article, such as links, categories and
info-boxes, see `ft.onto.wiki_ontology` for more information.

This reader is based on DBpedia's extracted datasets.
"""
import logging
from collections import defaultdict
from typing import Iterator, Dict, List, DefaultDict, Tuple

import rdflib

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.datasets.wikipedia.db_utils import (
    NIFParser, NIFBufferedContextReader, get_resource_attribute,
    get_resource_name, get_resource_fragment,
    print_progress)
from forte.data.readers.base_reader import PackReader
from ft.onto.wikipedia import (WikiPage, WikiSection, WikiParagraph, WikiTitle,
                               WikiAnchor, WikiInfoBoxMapped)

state_type = Tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]


def add_struct(pack: DataPack, struct_statements: List):
    for nif_range, rel, struct_type in struct_statements:
        r = get_resource_fragment(rel)
        if r == 'type':
            range_ = get_resource_attribute(nif_range, 'char')
            begin, end = [int(d) for d in range_.split(',')]

            if end > len(pack.text):
                # Some nif dataset are off by a bit, mostly when there are
                # new line characters, we cannot correct them.
                # but we need to make sure they don't go longer than the text.
                logging.info("NIF Structure end is %d by %s, "
                             "clipped to fit with the text.", end, nif_range)
                end = len(pack.text)

            if end <= begin:
                logging.info("Provided struct [%d:%d] is invalid.", begin, end)
                continue

            struct_ = get_resource_fragment(struct_type)

            if struct_ == 'Section':
                WikiSection(pack, begin, end)
            elif struct_ == 'Paragraph':
                WikiParagraph(pack, begin, end)
            elif struct_ == 'Title':
                WikiTitle(pack, begin, end)
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

        if end > len(pack.text):
            # Some nif dataset are off by a bit, mostly when there are
            # new line characters, we cannot correct them.
            # but we need to make sure they don't go longer than the text.
            logging.info("Provided anchor end is %d, "
                         "clipped to fit with the text.", end)
            end = len(pack.text)

        if end <= begin:
            logging.info("Provided anchor [%d:%d is invalid.]", begin, end)
            continue

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
                anchor.target_page_name = target_page_name


def add_info_boxes(pack: DataPack, info_box_statements: List):
    for _, v, o in info_box_statements:
        slot_name = v.toPython()
        slot_value = get_resource_name(o)
        info_box = WikiInfoBoxMapped(pack)
        info_box.key = slot_name
        info_box.value = slot_value


class DBpediaWikiReader(PackReader):
    def __init__(self):
        super().__init__()
        self.struct_reader = None
        self.link_reader = None
        self.redirects: Dict[str, str] = {}

    def initialize(self, resources: Resources, configs: Config):
        self.redirects = resources.get('redirects')

        # These NIF readers organize the statements in the specific RDF context,
        # in this case each context correspond to one wiki page, this allows
        # us to read the information more systematically.
        self.struct_reader = NIFBufferedContextReader(
            configs.nif_page_structure)
        self.link_reader = NIFBufferedContextReader(configs.nif_text_links)

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

        pack = self.new_pack()
        doc_name: str = str_data['doc_name']
        if doc_name in self.redirects:
            doc_name = self.redirects[doc_name]

        full_text: str = str_data['text']

        pack.set_text(full_text)
        page = WikiPage(pack, 0, len(full_text))
        page.page_id = str_data['oldid']
        page.page_name = doc_name

        if len(node_data['struct']) > 0:
            add_struct(pack, node_data['struct'])
        else:
            logging.warning('Structure info for %s not found.', doc_name)

        if len(node_data['links']) > 0:
            add_anchor_links(pack, node_data['links'], self.redirects)
        else:
            logging.warning('Links for [%s] not found.', doc_name)

        pack.pack_name = doc_name

        yield pack

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure
        :return:
        """
        return {
            'redirect_path': None,
            'nif_page_structure': None,
            'nif_text_links': None,
        }
