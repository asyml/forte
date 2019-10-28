import logging
import os
from typing import Dict, List, Tuple, Optional

import torch
import texar.torch as tx
from texar.torch.hyperparams import HParams

from forte.data.base import Span
from forte.data.data_pack import DataPack
from forte.common.resources import Resources
from forte.common.types import DataRequest
from forte.models.srl.model import LabeledSpanGraphNetwork
from forte.processors.base.batch_processor import FixedSizeBatchProcessor
from ft.onto.base_ontology import (
    Token, Sentence, PredicateLink, PredicateMention, PredicateArgument)

logger = logging.getLogger(__name__)

__all__ = [
    "SRLPredictor",
]

Prediction = List[
    Tuple[Span, List[Tuple[Span, str]]]
]


class SRLPredictor(FixedSizeBatchProcessor):
    """
    An Semantic Role labeler trained according to `He, Luheng, et al.
    "Jointly predicting predicates and arguments in neural semantic role
    labeling." <https://aclweb.org/anthology/P18-2058>`_.
    """

    word_vocab: tx.data.Vocab
    char_vocab: tx.data.Vocab
    model: LabeledSpanGraphNetwork

    def __init__(self):
        super().__init__()

        self.define_context()

        self.batch_size = 4
        self.batcher = self.define_batcher()

        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')

    def initialize(self,
                   _: Resources,
                   configs: Optional[HParams]):

        model_dir = configs.storage_path if configs is not None else None
        logger.info("restoring SRL model from %s", model_dir)

        self.word_vocab = tx.data.Vocab(
            os.path.join(model_dir, "embeddings/word_vocab.english.txt"))
        self.char_vocab = tx.data.Vocab(
            os.path.join(model_dir, "embeddings/char_vocab.english.txt"))
        model_hparams = LabeledSpanGraphNetwork.default_hparams()
        model_hparams["context_embeddings"]["path"] = os.path.join(
            model_dir, model_hparams["context_embeddings"]["path"])
        model_hparams["head_embeddings"]["path"] = os.path.join(
            model_dir, model_hparams["head_embeddings"]["path"])
        self.model = LabeledSpanGraphNetwork(
            self.word_vocab, self.char_vocab, model_hparams)
        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, "pretrained/model.pt"),
            map_location=self.device))
        self.model.eval()

    def define_context(self):
        self.context_type = Sentence

    # pylint: disable=no-self-use
    def _define_input_info(self) -> DataRequest:
        input_info: DataRequest = {Token: []}
        return input_info

    def predict(self, data_batch: Dict) -> Dict[str, List[Prediction]]:
        text: List[List[str]] = [
            sentence.tolist() for sentence in data_batch["Token"]["text"]]
        text_ids, length = tx.data.padded_batch([
            self.word_vocab.map_tokens_to_ids_py(sentence)
            for sentence in text])
        text_ids = torch.from_numpy(text_ids).to(device=self.device)
        length = torch.tensor(length, dtype=torch.long, device=self.device)
        batch_size = len(text)
        batch = tx.data.Batch(batch_size, text=text, text_ids=text_ids,
                              length=length, srl=[[]] * batch_size)
        self.model = self.model.cuda()
        batch_srl_spans = self.model.decode(batch)

        # Convert predictions into annotations.
        batch_predictions: List[Prediction] = []
        for idx, srl_spans in enumerate(batch_srl_spans):
            word_spans = data_batch["Token"]["span"][idx]
            predictions: Prediction = []
            for pred_idx, pred_args in srl_spans.items():
                begin, end = word_spans[pred_idx]
                # TODO cannot create annotation here.
                pred_span = Span(begin, end)
                arguments = []
                for arg in pred_args:
                    begin = word_spans[arg.start][0]
                    end = word_spans[arg.end][1]
                    arg_annotation = Span(begin, end)
                    arguments.append((arg_annotation, arg.label))
                predictions.append((pred_span, arguments))
            batch_predictions.append(predictions)
        return {"predictions": batch_predictions}

    def pack(self, data_pack: DataPack,
             inputs: Dict[str, List[Prediction]]) -> None:
        batch_predictions = inputs["predictions"]
        for predictions in batch_predictions:
            for pred_span, arg_result in predictions:

                pred = data_pack.add_entry(PredicateMention(
                    data_pack, pred_span.begin, pred_span.end)
                )

                for arg_span, label in arg_result:
                    arg = data_pack.add_or_get_entry(PredicateArgument(
                            data_pack, arg_span.begin, arg_span.end
                        )
                    )
                    link = PredicateLink(data_pack, pred, arg)
                    link.set_fields(arg_type=label)
                    data_pack.add_or_get_entry(link)

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure
        :return:
        """
        hparams_dict = {
            'storage_path': None,
        }
        return hparams_dict
