"""
Read wikipedia articles and parse each article into a DataPack, which contains
some basic structure of a Wikipedia article, such as links, categories and
info-boxes, see `forte.data.ontology.wiki_ontology` for more information.

This reader is based on DBpedia's extracted datasets.
"""
from typing import Any, Iterator, Dict, Type, Union, List

from forte import PackType
from forte.data.ontology import Entry
from forte.data.readers import PackReader
from forte.data.datasets.wikipedia.dbpedia_supports import NIFParser


class DBpediaWikiReader(PackReader):
    def define_output_info(self) -> Dict[Type[Entry], Union[List, Dict]]:
        pass

    def _collect(self, nif_context: str) -> Iterator[Any]:
        nif_context = NIFParser(nif_context)

        pass

    def parse_pack(self, collection: Any) -> PackType:
        pass

    def _cache_key_function(self, collection: Any) -> str:
        pass
