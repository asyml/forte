"""
Read the DBpedia info box datasets and add the content into the DataPacks.

This will use the following datasets from DBpedia:
        -- mappingbased_literals_en.tql.bz2
        -- mappingbased_objects_en.tql.bz2
        -- infobox_properties_wkd_uris_en.tql.bz2
"""
import csv
import os
from typing import Optional, List, Iterator, Dict, Type, Union

from texar.torch import HParams

from forte import Resources
from forte.data import DataPack
from forte.data.datasets.wikipedia.db_utils import (
    NIFParser, get_resource_name, NIFBufferedContextReader)
from forte.data.ontology import Entry
from forte.data.ontology.wiki_ontology import WikiInfoBoxEntry
from forte.data.readers import PackReader


def add_info_boxes(pack: DataPack, info_box_statements: List,
                   info_type: str):
    for _, v, o, _ in info_box_statements:
        slot_name = v.toPython()
        slot_value = get_resource_name(o)
        info_box = WikiInfoBoxEntry(pack)
        info_box.set_key(slot_name)
        info_box.set_value(slot_value)

        if info_type == 'literal':
            info_box.set_is_literal(True)
        elif info_type == 'object':
            info_box.set_is_object(True)
        elif info_type == 'raw':
            info_box.set_is_raw(True)

        pack.add_entry(info_box)


def read_index(pack_index_path: str) -> Dict[str, str]:
    page_idx: Dict[str, str] = {}

    with csv.reader(pack_index_path, delimiter='\t') as idx:
        for page_name, page_path in idx:
            page_idx[page_name] = page_path
    return page_idx


class DBpediaInfoBoxReader(PackReader):
    def initialize(self, resource: Resources, configs: Optional[HParams]):
        self.pack_index = read_index(configs.pack_index)
        self.pack_dir = configs.pack_dir

    def _collect(self, mapping_literals: str, mapping_objects: str,
                 info_box_raw: str) -> Iterator[Dict[str, str]]:
        last_con = None
        info_box_statements = []

        literal_info_reader = NIFBufferedContextReader(mapping_literals)
        object_info_reader = NIFBufferedContextReader(mapping_objects)

        for statements in NIFParser(info_box_raw):
            for resource, rel, value, c in statements:
                if not c == last_con and last_con is not None:
                    literals = literal_info_reader.get(c)
                    objects = object_info_reader.get(c)

                    yield {
                        'resource': get_resource_name(resource),
                        'raw': info_box_statements,
                        'literals': literals,
                        'objects': objects,
                    }
                info_box_statements.append((resource, rel, value))
                last_con = c

        if len(info_box_statements) > 0:
            literals = literal_info_reader.get(last_con)
            objects = object_info_reader.get(last_con)

            resource = info_box_statements[0][0]

            yield {
                'resource': get_resource_name(resource),
                'raw': info_box_statements,
                'literals': literals,
                'objects': objects,
            }

    def parse_pack(self, info_box_data: Dict[str, List]) -> DataPack:
        if info_box_data['resource'] in self.pack_index:
            pack_path = os.path.join(self.pack_dir,
                                     self.pack_index[info_box_data['resource']])

            if os.path.exists(pack_path):
                with open(pack_path) as pack_file:
                    pack = self.deserialize_instance(pack_file.read())

                    add_info_boxes(pack, info_box_data['raw'], 'raw')
                    add_info_boxes(pack, info_box_data['literals'], 'literal')
                    add_info_boxes(pack, info_box_data['objects'], 'object')

                    return pack

    def _cache_key_function(self, info_box_data: Dict[str, List]) -> str:
        pass

    def define_output_info(self) -> Dict[Type[Entry], Union[List, Dict]]:
        pass

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure
        :return:
        """
        return {
            'pack_index': 'article.idx',
            'pack_dir': '.',
        }
