"""
Read wikipedia articles and parse each article into a DataPack, which contains
some basic structure of a Wikipedia article, such as links, categories and
info-boxes, see `forte.data.ontology.wiki_ontology` for more information.

This reader is based on DBpedia's extracted datasets.
"""
from typing import Any, Iterator, Dict, Type, Union, List

from forte.data import DataPack
from forte.data.datasets.wikipedia import nif_utils
from forte.data.ontology import Entry
from forte.data.readers import PackReader
from forte.data.datasets.wikipedia.nif_utils import NIFParser, NIFContextReader


class DBpediaWikiReader(PackReader):
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

        struct_reader = NIFContextReader(nif_page_structure)

        for graph in NIFParser(nif_context):
            for statement in graph:
                s, v, o = statement
                nif_type = nif_utils.get_resource_attribute(s, "nif")

                if nif_type and nif_type == "context" and v.endswith(
                        "nif-core#isString"):
                    doc_data['context'] = str(o)
                    doc_data['doc_id'] = nif_utils.get_dbpedia_resource_name(s)

                    yield doc_data

    def parse_pack(self, doc_data: Dict[str, Any]) -> DataPack:
        pack = DataPack()
        pack.set_text(doc_data['context'])
        pack.meta.doc_id = doc_data['doc_id']
        return pack

    def _cache_key_function(self, collection: Any) -> str:
        pass
