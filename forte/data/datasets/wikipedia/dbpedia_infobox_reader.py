"""
Read the DBpedia info box datasets and add the content into the DataPacks.

This will use the following datasets from DBpedia:
        -- mappingbased_literals_en.tql.bz2
        -- mappingbased_objects_en.tql.bz2
        -- infobox_properties_wkd_uris_en.tql.bz2
"""
import csv
import os
from typing import List, Iterator, Dict, Tuple

from smart_open import open
from texar.torch import HParams
import rdflib

from forte import Resources, logging
from forte.data import DataPack
from forte.data.datasets.wikipedia.db_utils import (
    get_resource_name, NIFBufferedContextReader, ContextGroupedNIFReader,
    print_progress, print_notice)
from forte.data.readers import PackReader
from ft.onto.wikipedia import WikiInfoBoxProperty, WikiInfoBoxMapped

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
        info_box = WikiInfoBoxMapped(pack)
        info_box.set_key(v.toPython())
        info_box.set_value(get_resource_name(o))
        info_box.set_infobox_type(info_type)
        pack.add_entry(info_box)


def read_index(pack_index_path: str) -> Dict[str, str]:
    page_idx: Dict[str, str] = {}

    logging.info("Reading pack index from %s", pack_index_path)

    with open(pack_index_path) as idx:
        for page_name, page_path in csv.reader(idx, delimiter='\t'):
            page_idx[page_name] = page_path
    return page_idx


class DBpediaInfoBoxReader(PackReader):
    def __init__(self):
        super().__init__()
        self.pack_index: Dict[str, str]
        self.pack_dir: str
        self.redirects: Dict[str, str]
        self.logger = logging.getLogger(__name__)

    def initialize(self, resource: Resources, configs: HParams):
        # pylint: disable=attribute-defined-outside-init
        self.pack_index = read_index(configs.pack_index)
        self.pack_dir = configs.pack_dir

        self.redirects = resource.get('redirects')

        self.literal_info_reader = NIFBufferedContextReader(
            configs.mapping_literals)
        self.object_info_reader = NIFBufferedContextReader(
            configs.mapping_objects)

        # Set up logging.
        f_handler = logging.FileHandler(configs.reading_log)
        f_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)
        self.logger.handlers = [f_handler]

    def _collect(self, info_box_raw: str  # type: ignore
                 ) -> Iterator[Tuple[str, Dict[str, List[state_type]]]]:
        for c, statements in ContextGroupedNIFReader(info_box_raw):
            yield get_resource_name(statements[0][0]), {
                'properties': statements,
                'literals': self.literal_info_reader.get(c),
                'objects': self.object_info_reader.get(c),
            }

    def _parse_pack(
            self, collection: Tuple[str, Dict[str, List[state_type]]]
    ) -> Iterator[DataPack]:
        resource_name, info_box_data = collection

        if resource_name in self.redirects:
            resource_name = self.redirects[resource_name]

        if resource_name in self.pack_index:
            print_progress(f'Add infobox to resource: [{resource_name}]')

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
        else:
            print_notice(f"Resource {resource_name} is not in the raw packs.")
            self.logger.warning("Resource %s is not in the raw packs.",
                                resource_name)

    def _cache_key_function(self, info_box_data: Dict[str, List]) -> str:
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
            'mapping_literals': None,
            'mapping_objects': None,
            'reading_log': 'infobox.log',
        }
