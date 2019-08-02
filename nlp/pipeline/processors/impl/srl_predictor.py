import os
import logging
from typing import Dict, List, Tuple

import texar.torch as tx
import torch

from nlp.pipeline.common.resources import Resources
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.ontology import ontonotes_ontology, base_ontology
from nlp.pipeline.models.srl.model import LabeledSpanGraphNetwork
from nlp.pipeline.processors.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "SRLPredictor",
]

Prediction = Dict[
    ontonotes_ontology.PredicateMention,
    List[Tuple[ontonotes_ontology.PredicateArgument, str]]]


class SRLPredictor(BatchProcessor):
    word_vocab: tx.data.Vocab
    char_vocab: tx.data.Vocab
    model: LabeledSpanGraphNetwork

    def __init__(self, model_dir: str):
        super().__init__()

        self._ontology = ontonotes_ontology
        self.define_input_info()
        self.define_output_info()

        self.context_type = "sentence"

        self.batch_size = 4
        self.batcher = self.initialize_batcher()

        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')

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

    def define_input_info(self):
        self.input_info = {
            base_ontology.Token: [],
        }

    def define_output_info(self):
        self.output_info = {
            self._ontology.PredicateMention:
                ["pred_type", "span"],
            self._ontology.PredicateArgument: ["span"],
            self._ontology.PredicateLink:
                ["parent", "child", "arg_type"],

        }

    def initialize(self, resource: Resources):
        raise NotImplementedError

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
            predictions: Prediction = {}
            for pred_idx, pred_args in srl_spans.items():
                begin, end = word_spans[pred_idx]
                pred_annotation = self._ontology.PredicateMention(begin, end)
                arguments = []
                for arg in pred_args:
                    begin = word_spans[arg.start][0]
                    end = word_spans[arg.end][1]
                    arg_annotation = self._ontology.PredicateArgument(begin,
                                                                      end)
                    arguments.append((arg_annotation, arg.label))
                predictions[pred_annotation] = arguments
            batch_predictions.append(predictions)
        return {"predictions": batch_predictions}

    def pack(self, data_pack: DataPack,
             inputs: Dict[str, List[Prediction]]) -> None:
        batch_predictions = inputs["predictions"]
        for predictions in batch_predictions:
            for pred, args in predictions.items():
                pred = data_pack.add_or_get_entry(pred)
                for arg, label in args:
                    arg = data_pack.add_or_get_entry(arg)
                    link = self._ontology.PredicateLink(pred, arg)
                    link.set_fields(arg_type=label)
                    data_pack.add_or_get_entry(link)
