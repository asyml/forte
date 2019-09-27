import logging
from typing import Iterator, Optional, List

from texar.torch import HParams

from forte import config
from forte.common import Resources
from forte.indexers import EmbeddingBasedIndexer
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology import base_ontology
from forte.data.readers import MultiPackReader

logger = logging.getLogger(__name__)

__all__ = [
    "SimpleSearcher",
]


class SimpleSearcher(MultiPackReader):

    def __init__(self, index_path: Optional[str]):
        super().__init__()
        self._ontology = base_ontology
        self.index = EmbeddingBasedIndexer(hparams={
            "index_type": "GpuIndexFlatIP",
            "dim": 768
        })
        self.index.load(index_path)
        self.k = 5

    def initialize(self, resource: Resources, configs: HParams):
        self.resources = resource

    def define_output_info(self):
        return {}

    # pylint: disable=no-self-use,unused-argument
    def _cache_key_function(self, collection):
        return "cached_string_file"

    # pylint: disable=no-self-use
    def _collect(self, data: DataPack) -> Iterator[str]:
        """
        data_strings should be of type `List[str]`
        which is the list of raw text strings to iterate over
        """
        # This allows the user to pass in either one single string or a list of
        # strings.
        data_array = [data] if not isinstance(data, list) else data
        for data in data_array:
            query = data.query.query
            results = self.index.search(query, self.k)
            #import pdb
            #pdb.set_trace()
            documents = [r[1] for result in results for r in result]
            yield documents

    def parse_pack(self, documents: List[str]) -> MultiPack:
        """
        Takes a raw string and converts into a DataPack
        :param documents: str that contains text of a document
        :return: DataPack containing Document
        """
        config.working_component = self.component_name

        multi_pack = MultiPack()
        packs = {}
        import pdb
        pdb.set_trace()
        counter = 0
        for doc in documents:
            pack = DataPack()
            document = self._ontology.Document(0, len(doc))
            pack.add_entry(document)
            pack.set_text(doc)
            packs[f"doc_{counter}"] = pack
            counter += 1

        multi_pack.update_pack(**packs)

        config.working_component = None

        return multi_pack

