# pylint: disable=attribute-defined-outside-init
import torch
from texar.torch.hyperparams import HParams
from texar.torch.modules import BERTEncoder
from texar.torch.data import BERTTokenizer

from forte.common.resources import Resources
from forte.data import MultiPack
from forte.data.ontology import base_ontology
from forte.processors.base import MultiPackProcessor, ProcessInfo

__all__ = [
    "QueryCreator"
]


class QueryCreator(MultiPackProcessor):
    r"""This processor is used to search for relevant documents for a query
    """

    def __init__(self) -> None:
        super().__init__()
        self._ontology = base_ontology

    def initialize(self, resources: Resources, configs: HParams):
        self.resource = resources
        vocab_file = configs.vocab_file
        self.tokenizer = BERTTokenizer.load(vocab_file)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = BERTEncoder(pretrained_model_name="bert-base-uncased")
        self.encoder.to(self.device)

    # pylint: disable=no-self-use
    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {

        }

        return input_info

    # pylint: disable=no-self-use
    def _define_output_info(self) -> ProcessInfo:
        output_info: ProcessInfo = {

        }

        return output_info

    @torch.no_grad()
    def get_embeddings(self, input_ids, segment_ids):
        return self.encoder(inputs=input_ids, segment_ids=segment_ids)

    def _process(self, input_pack: MultiPack):
        input_ids = []
        segment_ids = []

        query_pack = input_pack.get_pack("pack")
        context = [query_pack.text]

        # use context to build the query
        if "user_utterance" in input_pack.pack_names:
            user_pack = input_pack.get_pack("user_utterance")
            context.append(user_pack.text)

        if "bot_utterance" in input_pack.pack_names:
            bot_pack = input_pack.get_pack("bot_utterance")
            context.append(bot_pack.text)

        for text in context:
            t = self.tokenizer.encode_text(text)
            input_ids.append(t[0])
            segment_ids.append(t[1])

        input_ids = torch.LongTensor(input_ids).to(self.device)
        segment_ids = torch.LongTensor(segment_ids).to(self.device)
        _, query_vector = self.get_embeddings(input_ids, segment_ids)
        query_vector = torch.mean(query_vector, dim=0, keepdim=True)
        query_vector = query_vector.cpu().numpy()
        query_pack.query = query_vector
