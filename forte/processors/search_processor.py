# pylint: disable=attribute-defined-outside-init
from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data import DataPack, MultiPack
from forte.data.ontology import Query
from forte.processors.base import MultiPackProcessor
from forte.indexers import EmbeddingBasedIndexer

from ft.onto.base_ontology import Document

__all__ = [
    "SearchProcessor"
]


class SearchProcessor(MultiPackProcessor):
    r"""This processor searches for relevant documents for a query"""

    def __init__(self) -> None:
        super().__init__()

        self.index = EmbeddingBasedIndexer(hparams={
            "index_type": "GpuIndexFlatIP",
            "dim": 768,
            "device": "gpu0"
        })

    def initialize(self, resources: Resources, configs: HParams):

        self.resources = resources
        self.config = configs
        self.index.load(self.config.model_dir)
        self.k = self.config.k or 5

    def _process(self, input_pack: MultiPack):
        query_pack = input_pack.get_pack(self.config.query_pack_name)
        first_query = list(query_pack.get_entries(Query))[0]
        results = self.index.search(first_query.value, self.k)
        documents = [r[1] for result in results for r in result]

        packs = {}
        for i, doc in enumerate(documents):
            pack = DataPack()
            document = Document(pack=pack, begin=0, end=len(doc))
            pack.add_entry(document)
            pack.set_text(doc)
            packs[self.config.response_pack_name[i]] = pack

        input_pack.update_pack(packs)
