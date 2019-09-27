from typing import Optional

import torch
from texar.torch.hyperparams import HParams
from texar.torch.modules import BERTEncoder
from texar.torch.data import BERTTokenizer

from forte.common.resources import Resources
from forte.data import DataPack
from forte.data.ontology import base_ontology
from forte.data.ontology import Query
from forte.processors.base import PackProcessor, ProcessInfo

__all__ = [
    "QueryCreator",
]


class QueryCreator(PackProcessor):
    r"""This processor is used to search for relevant documents for a query
    """

    def __init__(self) -> None:
        super().__init__()
        self._ontology = base_ontology

    def initialize(self, configs: HParams,
                   resouces: Optional[Resources] = None):

        # create a BERT tokenizer
        vocab_file = configs["vocab_file"]
        self.tokenizer = BERTTokenizer.load(vocab_file)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = BERTEncoder(pretrained_model_name="bert-base-uncased")
        self.encoder.to(self.device)

    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {
        }

        return input_info

    def _define_output_info(self) -> ProcessInfo:
        output_info: ProcessInfo = {

        }
        return output_info

    @torch.no_grad()
    def get_embeddings(self, input_ids, segment_ids):
        return self.encoder(inputs=input_ids, segment_ids=segment_ids)

    def _process(self, input_pack: DataPack):
        input_ids = []
        segment_ids = []
        query = list(input_pack.get(self._ontology.Document))[0]
        t = self.tokenizer.encode_text(query.text)
        input_ids.append(t[0])
        segment_ids.append(t[1])
        input_ids = torch.LongTensor(input_ids).to(self.device)
        segment_ids = torch.LongTensor(segment_ids).to(self.device)
        _, query_vector = self.get_embeddings(input_ids, segment_ids)
        query_vector = query_vector.cpu().numpy()
        input_pack.query = Query(query=query_vector)
