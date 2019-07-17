import os
import logging
from typing import Dict, List, Tuple

import texar as tx
import torch

from nlp.pipeline.common.resources import Resources
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.readers import OntonotesOntology
from nlp.pipeline.models.srl.model import LabeledSpanGraphNetwork
from nlp.pipeline.processors.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "SRLPredictor",
]

Prediction = Dict[
    OntonotesOntology.PredicateMention,
    List[Tuple[OntonotesOntology.PredicateArgument, str]]]


class SRLPredictor(BatchProcessor):
    word_vocab: tx.data.Vocab
    char_vocab: tx.data.Vocab
    model: LabeledSpanGraphNetwork

    def __init__(self, model_dir: str):
        super().__init__()

        self.component_name = "srl_predictor"
        self.context_type = "sentence"
        self.annotation_types = {
            "Token": [],
        }
        self.batch_size = 4
        self.initialize_batcher()

        self.ontology = OntonotesOntology
        self.device = (torch.device(torch.cuda.current_device())
                       if torch.cuda.is_available() else 'cpu')

        logger.info("restoring SRL model from {}".format(model_dir))

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

    def initialize(self, resource: Resources):
        raise NotImplementedError

    def _record_fields(self, data_pack: DataPack):
        data_pack.record_fields(
            ["pred_type", "span"],
            self.component_name,
            self.ontology.PredicateMention.__name__,
        )
        data_pack.record_fields(
            ["span"],
            self.component_name,
            self.ontology.PredicateArgument.__name__,
        )
        data_pack.record_fields(
            ["parent", "child", "arg_type"],
            self.component_name,
            self.ontology.PredicateLink.__name__,
        )

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
                pred_annotation = OntonotesOntology.PredicateMention(
                    self.component_name, begin, end)
                arguments = []
                for arg in pred_args:
                    begin = word_spans[arg.start][0]
                    end = word_spans[arg.end][1]
                    arg_annotation = OntonotesOntology.PredicateArgument(
                        self.component_name, begin, end)
                    arguments.append((arg_annotation, arg.label))
                predictions[pred_annotation] = arguments
            batch_predictions.append(predictions)
        return {"predictions": batch_predictions}

    def pack(self, data_pack: DataPack,
             inputs: Dict[str, List[Prediction]]) -> None:
        batch_predictions = inputs["predictions"]
        for predictions in batch_predictions:
            for pred, args in predictions.items():
                pred_id = data_pack.add_entry(pred)
                for arg, label in args:
                    arg_id = data_pack.add_entry(arg)
                    link = OntonotesOntology.PredicateLink(
                        self.component_name, pred_id, arg_id)
                    link.set_fields(arg_type=label)
                    data_pack.add_entry(link)
