# pylint: disable=attribute-defined-outside-init
from typing import Optional

from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data import DataPack, MultiPack
from forte.data.ontology import Query
from forte.processors.base import MultiPackProcessor
from forte.indexers import EmbeddingBasedIndexer

from ft.onto.base_ontology import Document

__all__ = [
    "SearchProcessor",
]


class SearchProcessor(MultiPackProcessor):
    r"""This processor is used to search for relevant documents for a query
    """

    def __init__(self) -> None:
        super().__init__()

        self.index = EmbeddingBasedIndexer(hparams={
            "index_type": "GpuIndexFlatIP",
            "dim": 768,
            "device": "gpu0"
        })

    def initialize(self, resources: Optional[Resources],
                   configs: Optional[HParams]):

        self.resources = resources

        if configs:
            self.index.load(configs.model_dir)
            self.k = configs.k or 5

    def _process(self, input_pack: MultiPack):
        query_pack = input_pack.get_pack("pack")
        first_query = list(query_pack.get_entries(Query))[0]
        results = self.index.search(first_query.value, self.k)
        documents = [r[1] for result in results for r in result]

        packs = {}
        counter = 0
        for doc in documents:
            pack = DataPack()
            document = Document(pack=pack, begin=0, end=len(doc))
            pack.add_entry(document)
            pack.set_text(doc)
            packs[f"doc_{counter}"] = pack
            counter += 1

        input_pack.update_pack(packs)
