"""
Read the DBpedia info box datasets and add the content into the DataPacks.

This will use the following datasets from DBpedia:
        -- mappingbased_literals_en.tql.bz2
        -- mappingbased_objects_en.tql.bz2
        -- infobox_properties_wkd_uris_en.tql.bz2
"""
import csv
import os
from typing import Optional, List, Iterator, Dict, Type, Union, Tuple

from smart_open import open
import rdflib
from texar.torch import HParams

from forte import Resources
from forte.data import DataPack
from forte.data.datasets.wikipedia.db_utils import (
    NIFParser, get_resource_name, NIFBufferedContextReader, context_base)
from forte.data.ontology import Entry
from forte.data.ontology.wiki_ontology import (
    WikiInfoBoxMapped, WikiInfoBoxProperty)
from forte.data.readers import PackReader

state_type = Tuple[rdflib.term.Node, rdflib.term.Node, rdflib.term.Node]


def add_property(pack: DataPack, statements: List):
    for _, v, o in statements:
        slot_name = v.toPython()
        slot_value = get_resource_name(o)
        info_box = WikiInfoBoxProperty(pack)
        info_box.set_key(slot_name)
        info_box.set_value(slot_value)
        pack.add_entry(info_box)


def add_info_boxes(pack: DataPack, info_box_statements: List,
                   info_type: str):
    for _, v, o in info_box_statements:
        slot_name = v.toPython()
        slot_value = get_resource_name(o)
        info_box = WikiInfoBoxMapped(pack)
        info_box.set_key(slot_name)
        info_box.set_value(slot_value)

        if info_type == 'literal':
            info_box.set_is_literal(True)
        elif info_type == 'object':
            info_box.set_is_object(True)

        pack.add_entry(info_box)


def read_index(pack_index_path: str) -> Dict[str, str]:
    page_idx: Dict[str, str] = {}

    with open(pack_index_path) as idx:
        for page_name, page_path in csv.reader(idx, delimiter='\t'):
            page_idx[page_name] = page_path
    return page_idx


class DBpediaInfoBoxReader(PackReader):
    def initialize(self, resource: Resources, configs: Optional[HParams]):
        # pylint: disable=attribute-defined-outside-init
        self.pack_index = read_index(configs.pack_index)  # type: ignore
        self.pack_dir = configs.pack_dir  # type: ignore
        self.pack_format = configs.pack_format  # type: ignore

    def _collect(self,  # type: ignore
                 mapping_literals: str, mapping_objects: str,
                 info_box_raw: str
                 ) -> Iterator[Tuple[str, Dict[str, List[state_type]]]]:
        last_con = None
        property_statements: List[state_type] = []

        literal_info_reader = NIFBufferedContextReader(mapping_literals)
        object_info_reader = NIFBufferedContextReader(mapping_objects)

        for statements in NIFParser(info_box_raw):
            for resource, rel, value, c in statements:
                if len(property_statements) > 0 and not context_base(
                        c) == context_base(last_con):
                    yield get_resource_name(property_statements[0][0]), {
                        'properties': property_statements,
                        'literals': literal_info_reader.get(last_con),
                        'objects': object_info_reader.get(last_con),
                    }
                    property_statements = []

                property_statements.append((resource, rel, value))
                last_con = c

        if len(property_statements) > 0:
            resource = property_statements[0][0]

            yield get_resource_name(resource), {
                'properties': property_statements,
                'literals': literal_info_reader.get(last_con),
                'objects': object_info_reader.get(last_con),
            }

    def parse_pack(self, collection: Tuple[str, Dict[str, List[state_type]]]
                   ) -> Iterator[DataPack]:
        resource_name, info_box_data = collection

        if resource_name in self.pack_index:
            pack_path = os.path.join(
                self.pack_dir,
                self.pack_index[resource_name]
            )

            if os.path.exists(pack_path):
                with open(pack_path) as pack_file:
                    pack = self.deserialize_instance(pack_file.read())

                    add_info_boxes(pack, info_box_data['literals'], 'literal')
                    add_info_boxes(pack, info_box_data['objects'], 'object')
                    add_property(pack, info_box_data['properties'])

                    yield pack

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
            'pack_format': 'gz',
        }
